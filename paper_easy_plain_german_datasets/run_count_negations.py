import re
import pandas as pd
import os
from pathlib import Path
from statistics import mean

import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from typing import Mapping
import utils as utils_mod
from utils import process_folder
from constants import (
    TIGER_ORIG,
    ASGC_ORIG,
    ASGC_PLAIN,
    ASGC_EASY,
    G4A_ORIG,
    G4A_PLAIN,
    DEplain_ORIG,
    DEplain_PLAIN,
    Leiko_PLAIN
)

# ---------------------------------------------------------------------
# Negation counting (regex-based, includes common inflected forms)
# ---------------------------------------------------------------------

# Negation counting logic now lives inside FE_NegationCount below
# (kept as a class to match py_lift's FEL_BaseExtractor architecture).


from py_lift.extractors import FEL_BaseExtractor
from typing import Optional


class FE_NegationCount(FEL_BaseExtractor):
    """
    Counts negation words (incl. common inflected forms) in a document.

    Unlike FEL_AnnotationCounter, this extractor does not count annotations
    of a given type directly -- it reassembles the document text from the
    Token annotations (since py_lift CAS objects expose text via Token
    spans, not via cas.sofa_string in this setup) and applies a regex-based
    negation count on that reconstructed text.
    """

    NEGATION_PATTERNS = [
        r"\bnicht\b",
        r"\bnein\b",
        r"\bkein(?:e|er|em|en|es)?\b",
        r"\bnichts\b",
        r"\bohne\b",
        r"\bnie\b",
        r"\bniemand(?:em|en)?\b",
    ]

    def __init__(self, strict: Optional[bool] = None):
        super().__init__(strict=strict)

    def feature_name(self) -> str:
        return "Negation_COUNT"

    def _get_text(self, cas) -> str:
        """
        Reconstructs the document text by joining the covered text of all
        Token annotations. This avoids relying on cas.sofa_string, which
        is not reliably populated in this pipeline.
        """
        tokens = [tok.get_covered_text() for tok in cas.select("Token")]
        return " ".join(tokens)

    def count(self, cas) -> int:
        text = self._get_text(cas)
        total_count = 0
        for pattern in self.NEGATION_PATTERNS:
            matches = re.findall(pattern, text, flags=re.IGNORECASE)
            total_count += len(matches)
        return total_count

    def extract(self, cas) -> bool:
        count = self.count(cas)
        self._add_feature(cas, self.feature_name(), count)
        return True


FEATURE_NAMES = ["Negation_COUNT"]


def negation_extractor(cas) -> None:
    """Thin wrapper so process_folder's feature_extractor signature matches
    the existing call sites (extractor(cas)), same as e.g.
    `FE_TokensPerSentence().extract(cas)` in the original script."""
    FE_NegationCount().extract(cas)


# ---------------------------------------------------------------------
# Plotting (same style as _plot_sentence_lengths, adapted for counts)
# ---------------------------------------------------------------------

def _plot_negation_counts(corpus_scores: dict, title: str, output_path: Path) -> None:
    """Create a violin plot with a swarm overlay for negation counts per document.

    Args:
        corpus_scores: Mapping from corpus label to a list of negation counts (one per document).
        title: Plot title.
        output_path: Destination file path for the PNG image.
    """
    df = pd.DataFrame({
        "corpus": sum([[name] * len(scores) for name, scores in corpus_scores.items()], []),
        "score": sum([list(scores) for scores in corpus_scores.values()], []),
    })

    def _category(name: str) -> str:
        upper = name.upper()
        if "ORIG" in upper:
            return "orig"
        if "EASY" in upper:
            return "easy"
        return "plain"

    df["category"] = df["corpus"].apply(_category)

    sns.set(style="whitegrid")
    plt.figure(figsize=(10, 6))
    palette = {"orig": "tab:blue", "plain": "tab:orange", "easy": "tab:green"}
    sns.violinplot(
        x="corpus",
        y="score",
        hue="category",
        data=df,
        inner="quartile",
        palette=palette,
        legend=False,
    )
    plt.title(title)
    plt.ylabel("Negation Count per Document")
    plt.xlabel("Corpus")
    ax = plt.gca()
    plt.setp(ax.get_xticklabels(), rotation=45, ha="center")
    fig = plt.gcf()
    fig.subplots_adjust(bottom=0.35)
    plt.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def main() -> None:
    extractor = negation_extractor
    feature_names = FEATURE_NAMES

    scores_tiger_orig = process_folder(TIGER_ORIG, feature_extractor=extractor, feature_names=feature_names)

    scores_crawled_orig = process_folder(ASGC_ORIG, feature_extractor=extractor, feature_names=feature_names)
    scores_crawled_plain = process_folder(ASGC_PLAIN, feature_extractor=extractor, feature_names=feature_names)
    scores_crawled_easy = process_folder(ASGC_EASY, feature_extractor=extractor, feature_names=feature_names)

    scores_g4a_orig = process_folder(G4A_ORIG, feature_extractor=extractor, feature_names=feature_names)
    scores_g4a_plain = process_folder(G4A_PLAIN, feature_extractor=extractor, feature_names=feature_names)

    scores_deplain_orig = process_folder(DEplain_ORIG, feature_extractor=extractor, feature_names=feature_names)
    scores_deplain_plain = process_folder(DEplain_PLAIN, feature_extractor=extractor, feature_names=feature_names)

    scores_leiko_plain = process_folder(Leiko_PLAIN, feature_extractor=extractor, feature_names=feature_names)

    # ---------------------------------------------------------------------
    # Compute the average negation count for each corpus.
    # ---------------------------------------------------------------------
    def compute_avg(scores_dict: dict[str, list[float]]) -> dict[str, float]:
        return {name: (mean(scores_dict.get(name, [])) if scores_dict.get(name) else float('nan')) for name in feature_names}

    avg_by_corpus: dict[str, dict[str, float]] = {
        "TIGER_ORIG": compute_avg(scores_tiger_orig),
        "ASGC_ORIG": compute_avg(scores_crawled_orig),
        "DEplain_ORIG": compute_avg(scores_deplain_orig),
        "G4A_ORIG": compute_avg(scores_g4a_orig),
        "ASGC_PLAIN": compute_avg(scores_crawled_plain),
        "DEplain_PLAIN": compute_avg(scores_deplain_plain),
        "G4A_PLAIN": compute_avg(scores_g4a_plain),
        "ASGC_EASY": compute_avg(scores_crawled_easy),
        "Leiko_PLAIN": compute_avg(scores_leiko_plain),
    }

    print("Average negation count per corpus:")
    for corpus_name, avg_scores in avg_by_corpus.items():
        print(f"  {corpus_name}: {avg_scores[feature_names[0]]:.3f}")

    # Plot violin + swarm for negation count distribution across corpora.
    _plot_negation_counts(
        {
            "TIGER_ORIG": scores_tiger_orig.get(feature_names[0], []),
            "ASGC_ORIG": scores_crawled_orig.get(feature_names[0], []),
            "ASGC_PLAIN": scores_crawled_plain.get(feature_names[0], []),
            "ASGC_EASY": scores_crawled_easy.get(feature_names[0], []),
            "DEplain_ORIG": scores_deplain_orig.get(feature_names[0], []),
            "DEplain_PLAIN": scores_deplain_plain.get(feature_names[0], []),
            "G4A_ORIG": scores_g4a_orig.get(feature_names[0], []),
            "G4A_PLAIN": scores_g4a_plain.get(feature_names[0], []),
            "Leiko_PLAIN": scores_leiko_plain.get(feature_names[0], []),
        },
        "Negation Count Distribution per Document",
        Path("output") / "negation_count.png",
    )


if __name__ == "__main__":
    main()