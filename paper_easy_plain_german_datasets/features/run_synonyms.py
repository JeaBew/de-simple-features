"""
Nomen-Konsistenzprüfung für Leichte-Sprache-Texte
====================================================

Prüft, ob im Text mehrere unterschiedliche Nomen für dasselbe Konzept
verwendet werden.

Kombiniert drei Signale:
  1. OpenThesaurus-Abgleich
  2. spaCy-Word-Embeddings
  3. BERT-Kontextvektoren

Voraussetzungen:
    pip install spacy requests torch transformers
    python -m spacy download de_core_news_lg
"""

from __future__ import annotations

import sys
import time
from dataclasses import dataclass, field

import requests
import spacy

import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer


OPENTHESAURUS_URL = "https://www.openthesaurus.de/synonyme/search"

SIMILARITY_THRESHOLD = 0.75
BERT_SIMILARITY_THRESHOLD = 0.82

REQUEST_DELAY = 0.3

DEFAULT_BERT_MODEL = "bert-base-german-cased"
# Alternativen:
# DEFAULT_BERT_MODEL = "deepset/gbert-base"
# DEFAULT_BERT_MODEL = "dbmdz/bert-base-german-cased"


@dataclass
class NounContext:
    sentence_text: str
    start_char: int
    end_char: int
    surface: str


@dataclass
class NounOccurrence:
    lemma: str
    surface_forms: set[str] = field(default_factory=set)
    count: int = 0
    sentence_indices: list[int] = field(default_factory=list)
    contexts: list[NounContext] = field(default_factory=list)


@dataclass
class SynonymCandidate:
    lemma_a: str
    lemma_b: str
    source: str
    similarity: float | None = None
    spacy_similarity: float | None = None
    bert_similarity: float | None = None

    def __str__(self) -> str:
        parts = [self.source]

        if self.spacy_similarity is not None:
            parts.append(f"spaCy={self.spacy_similarity:.2f}")

        if self.bert_similarity is not None:
            parts.append(f"BERT={self.bert_similarity:.2f}")

        if (
            self.spacy_similarity is None
            and self.bert_similarity is None
            and self.similarity is not None
        ):
            parts.append(f"Ähnlichkeit={self.similarity:.2f}")

        info = ", ".join(parts)
        return f"'{self.lemma_a}' <-> '{self.lemma_b}'  [{info}]"


_nlp = None
_bert_tokenizer = None
_bert_model = None
_bert_model_name = None


def get_nlp():
    """Lazy-Loading des spaCy-Modells."""
    global _nlp
    if _nlp is None:
        _nlp = spacy.load("de_core_news_lg")
    return _nlp


def get_bert(model_name: str = DEFAULT_BERT_MODEL):
    """Lazy-Loading von BERT-Modell und Tokenizer."""
    global _bert_tokenizer, _bert_model, _bert_model_name

    if _bert_model is None or _bert_model_name != model_name:
        tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)

        if not tokenizer.is_fast:
            raise RuntimeError(
                "Für die BERT-Kontextvektoren wird ein FastTokenizer benötigt, "
                "weil Offset-Mapping verwendet wird."
            )

        model = AutoModel.from_pretrained(model_name)
        model.eval()

        _bert_tokenizer = tokenizer
        _bert_model = model
        _bert_model_name = model_name

    return _bert_tokenizer, _bert_model


def extract_nouns(text: str) -> dict[str, NounOccurrence]:
    """Extrahiert alle Nomen mit Lemma, Formen und Kontextinformationen."""
    nlp = get_nlp()
    doc = nlp(text)
    occurrences: dict[str, NounOccurrence] = {}

    for sent_idx, sent in enumerate(doc.sents):
        sentence_text = sent.text
        sent_start = sent.start_char

        for tok in sent:
            if tok.pos_ != "NOUN":
                continue

            lemma = tok.lemma_
            occ = occurrences.setdefault(lemma, NounOccurrence(lemma=lemma))

            local_start = tok.idx - sent_start
            local_end = local_start + len(tok.text)

            occ.surface_forms.add(tok.text)
            occ.count += 1
            occ.sentence_indices.append(sent_idx)
            occ.contexts.append(
                NounContext(
                    sentence_text=sentence_text,
                    start_char=local_start,
                    end_char=local_end,
                    surface=tok.text,
                )
            )

    return occurrences


def get_synonyms_openthesaurus(word: str, timeout: float = 5.0) -> set[str]:
    """Fragt OpenThesaurus nach Synonymen eines Wortes ab."""
    try:
        r = requests.get(
            OPENTHESAURUS_URL,
            params={"q": word, "format": "application/json"},
            timeout=timeout,
        )
        r.raise_for_status()
        data = r.json()
    except (requests.RequestException, ValueError) as e:
        print(
            f"  [Warnung] OpenThesaurus-Abfrage für '{word}' fehlgeschlagen: {e}",
            file=sys.stderr,
        )
        return set()

    synonyms = set()

    for synset in data.get("synsets", []):
        for term in synset.get("terms", []):
            synonyms.add(term["term"] if "term" in term else term.get("word", ""))

    synonyms.discard(word)
    synonyms.discard("")

    return synonyms


@torch.inference_mode()
def bert_context_vector(
    sentence: str,
    start_char: int,
    end_char: int,
    model_name: str = DEFAULT_BERT_MODEL,
) -> torch.Tensor | None:
    """
    Berechnet einen BERT-Kontextvektor für genau ein Nomen im Satz.

    Vorgehen:
    - Satz tokenisieren
    - über Offset-Mapping die Subword-Tokens finden, die zum Nomen gehören
    - letzte BERT-Schicht für diese Subword-Tokens mitteln
    """
    tokenizer, model = get_bert(model_name)

    encoded = tokenizer(
        sentence,
        return_tensors="pt",
        return_offsets_mapping=True,
        truncation=True,
        max_length=256,
    )

    offsets = encoded.pop("offset_mapping")[0]

    outputs = model(**encoded)
    hidden = outputs.last_hidden_state[0]

    token_indices = []

    for idx, offset in enumerate(offsets):
        token_start = int(offset[0])
        token_end = int(offset[1])

        # Spezialtokens wie [CLS], [SEP] haben häufig Offset (0, 0)
        if token_start == token_end == 0:
            continue

        # Überlappung zwischen Token-Span und Nomen-Span
        overlaps = token_end > start_char and token_start < end_char

        if overlaps:
            token_indices.append(idx)

    if not token_indices:
        return None

    vec = hidden[token_indices].mean(dim=0)

    # Normieren, damit Cosine Similarity stabiler ist
    vec = F.normalize(vec, p=2, dim=0)

    return vec


def compute_bert_lemma_vectors(
    nouns: dict[str, NounOccurrence],
    model_name: str = DEFAULT_BERT_MODEL,
    max_contexts_per_lemma: int = 8,
) -> dict[str, torch.Tensor]:
    """
    Berechnet pro Lemma einen durchschnittlichen BERT-Kontextvektor.

    Wenn ein Lemma mehrfach vorkommt, werden mehrere Kontextvektoren gemittelt.
    max_contexts_per_lemma begrenzt die Laufzeit.
    """
    lemma_vectors: dict[str, torch.Tensor] = {}

    for lemma, occ in nouns.items():
        vectors = []

        for ctx in occ.contexts[:max_contexts_per_lemma]:
            vec = bert_context_vector(
                ctx.sentence_text,
                ctx.start_char,
                ctx.end_char,
                model_name=model_name,
            )

            if vec is not None:
                vectors.append(vec)

        if vectors:
            lemma_vec = torch.stack(vectors).mean(dim=0)
            lemma_vec = F.normalize(lemma_vec, p=2, dim=0)
            lemma_vectors[lemma] = lemma_vec

    return lemma_vectors


def cosine_similarity_torch(vec_a: torch.Tensor, vec_b: torch.Tensor) -> float:
    """Cosine Similarity zweier bereits normierter Vektoren."""
    return float(torch.dot(vec_a, vec_b).item())


def find_synonym_candidates(
    nouns: dict[str, NounOccurrence],
    use_thesaurus: bool = True,
    use_embeddings: bool = True,
    use_bert: bool = False,
    similarity_threshold: float = SIMILARITY_THRESHOLD,
    bert_similarity_threshold: float = BERT_SIMILARITY_THRESHOLD,
    bert_model_name: str = DEFAULT_BERT_MODEL,
) -> list[SynonymCandidate]:
    """Prüft alle Nomen-Paare auf mögliche Synonymie."""
    nlp = get_nlp()
    lemmas = list(nouns.keys())
    candidates: list[SynonymCandidate] = []

    thesaurus_cache: dict[str, set[str]] = {}

    if use_thesaurus:
        for lemma in lemmas:
            thesaurus_cache[lemma] = get_synonyms_openthesaurus(lemma)
            time.sleep(REQUEST_DELAY)

    bert_vectors: dict[str, torch.Tensor] = {}

    if use_bert:
        bert_vectors = compute_bert_lemma_vectors(
            nouns,
            model_name=bert_model_name,
        )

    for i, lemma_a in enumerate(lemmas):
        for lemma_b in lemmas[i + 1:]:
            sources = []

            is_thesaurus_match = use_thesaurus and (
                lemma_b in thesaurus_cache.get(lemma_a, set())
                or lemma_a in thesaurus_cache.get(lemma_b, set())
            )

            if is_thesaurus_match:
                sources.append("thesaurus")

            spacy_sim = None
            is_embedding_match = False

            if use_embeddings:
                doc_a, doc_b = nlp(lemma_a), nlp(lemma_b)

                if doc_a.has_vector and doc_b.has_vector:
                    spacy_sim = doc_a.similarity(doc_b)
                    is_embedding_match = spacy_sim >= similarity_threshold

            if is_embedding_match:
                sources.append("embedding")

            bert_sim = None
            is_bert_match = False

            if use_bert:
                vec_a = bert_vectors.get(lemma_a)
                vec_b = bert_vectors.get(lemma_b)

                if vec_a is not None and vec_b is not None:
                    bert_sim = cosine_similarity_torch(vec_a, vec_b)
                    is_bert_match = bert_sim >= bert_similarity_threshold

            if is_bert_match:
                sources.append("bert")

            if sources:
                source = "+".join(sources)

                # Für Rückwärtskompatibilität: similarity bevorzugt spaCy,
                # sonst BERT, sonst None.
                general_similarity = spacy_sim if spacy_sim is not None else bert_sim

                candidates.append(
                    SynonymCandidate(
                        lemma_a=lemma_a,
                        lemma_b=lemma_b,
                        source=source,
                        similarity=general_similarity,
                        spacy_similarity=spacy_sim,
                        bert_similarity=bert_sim,
                    )
                )

    return candidates


def check_text(
    text: str,
    use_thesaurus: bool = True,
    use_embeddings: bool = True,
    use_bert: bool = False,
    similarity_threshold: float = SIMILARITY_THRESHOLD,
    bert_similarity_threshold: float = BERT_SIMILARITY_THRESHOLD,
    bert_model_name: str = DEFAULT_BERT_MODEL,
) -> dict:
    """Führt die vollständige Prüfung durch und gibt einen Report als dict zurück."""
    nouns = extract_nouns(text)

    candidates = find_synonym_candidates(
        nouns,
        use_thesaurus=use_thesaurus,
        use_embeddings=use_embeddings,
        use_bert=use_bert,
        similarity_threshold=similarity_threshold,
        bert_similarity_threshold=bert_similarity_threshold,
        bert_model_name=bert_model_name,
    )

    return {
        "nouns": nouns,
        "candidates": candidates,
        "num_nouns": len(nouns),
        "num_candidates": len(candidates),
    }


def print_report(report: dict) -> None:
    print(f"\n{'=' * 60}")
    print(f"Gefundene Nomen (Lemmata): {report['num_nouns']}")
    print(f"Mögliche Synonym-Inkonsistenzen: {report['num_candidates']}")
    print(f"{'=' * 60}\n")

    if not report["candidates"]:
        print("Keine Kandidaten gefunden - Benennung scheint konsistent.")
        return

    for cand in report["candidates"]:
        occ_a = report["nouns"][cand.lemma_a]
        occ_b = report["nouns"][cand.lemma_b]

        print(f"⚠  {cand}")
        print(f"   '{cand.lemma_a}': {occ_a.count}x, Formen: {occ_a.surface_forms}")
        print(f"   '{cand.lemma_b}': {occ_b.count}x, Formen: {occ_b.surface_forms}")
        print()


if __name__ == "__main__":
    text = (
        "Maria kauft ein neues Fahrrad. "
        "Das Rad hat sieben Gänge. "
        "Sie fährt jeden Tag mit dem Zweirad zur Arbeit. "
        "Der Wagen von Maria steht in der Garage. "
        "Das Auto ist blau. "
        "Sie nutzt den Wagen nur am Wochenende. "
        "Ihr Hund heißt Bello. "
        "Der Vierbeiner liebt lange Spaziergänge. "
        "Am Sonntag geht sie mit dem Hund in den Park."
    )

    report = check_text(
        text,
        use_thesaurus=True,
        use_embeddings=True,
        use_bert=True,
        bert_similarity_threshold=0.82,
    )

    print_report(report)