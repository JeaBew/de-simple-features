import pandas as pd
import os
from pathlib import Path
from statistics import mean

import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from typing import Mapping
import utils as utils_mod

def _plot_sentence_lengths(corpus_scores: dict, title: str, output_path: Path) -> None:
    """Create a violin plot with a swarm overlay for sentence length scores.

    Args:
        corpus_scores: Mapping from corpus label to a list of token‑per‑sentence scores.
        title: Plot title.
        output_path: Destination file path for the PNG image.
    """
    # Build a DataFrame suitable for seaborn.
    df = pd.DataFrame({
        "corpus": sum([[name] * len(scores) for name, scores in corpus_scores.items()], []),
        "score": sum([list(scores) for scores in corpus_scores.values()], []),
    })

    sns.set(style="whitegrid")
    plt.figure(figsize=(10, 6))
    # Violin plot with muted palette for consistency.
    sns.violinplot(x="corpus", y="score", hue="corpus", data=df, inner="quartile", palette="muted", legend=False)
    plt.title(title)
    plt.ylabel("Tokens per Sentence")
    plt.xlabel("Corpus")
    # Rotate x‑axis labels 45° for readability.
    ax = plt.gca()
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    plt.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300)
    plt.close()

from utils import process_folder
import utils as utils_mod
from constants import (
    CRAWLED_ORIG,
    CRAWLED_PLAIN,
    CRAWLED_EASY,
    GFA_ORIG,
    GFA_PLAIN,
    DEPLAIN_ORIG,
    DEPLAIN_PLAIN,
    Leiko_PLAIN,
)
from py_lift.extractors import FE_TokensPerSentence

def main() -> None:
    extractor = lambda cas: (
        FE_TokensPerSentence().extract(cas),
    )
    feature_names = [
            'Token_COUNT_PER_Sentence_COUNT'
    ]

    scores_crawled_orig = process_folder(CRAWLED_ORIG, feature_extractor=extractor, feature_names=feature_names)
    scores_crawled_plain = process_folder(CRAWLED_PLAIN, feature_extractor=extractor, feature_names=feature_names)
    scores_crawled_easy = process_folder(CRAWLED_EASY, feature_extractor=extractor, feature_names=feature_names)

    scores_g4a_orig = process_folder(GFA_ORIG, feature_extractor=extractor, feature_names=feature_names)
    scores_g4a_plain = process_folder(GFA_PLAIN, feature_extractor=extractor, feature_names=feature_names)

    scores_deplain_orig = process_folder(DEPLAIN_ORIG, feature_extractor=extractor, feature_names=feature_names)
    scores_deplain_plain = process_folder(DEPLAIN_PLAIN, feature_extractor=extractor, feature_names=feature_names)
    
    scores_leiko_plain = process_folder(Leiko_PLAIN, feature_extractor=extractor, feature_names=feature_names)
    
    # ---------------------------------------------------------------------
    # Compute the average score for each feature across all documents.
    # ---------------------------------------------------------------------
    # Compute average scores per feature for each corpus.
    def compute_avg(scores_dict: dict[str, list[float]]) -> dict[str, float]:
        return {name: (mean(scores_dict.get(name, [])) if scores_dict.get(name) else float('nan')) for name in feature_names}

    avg_by_corpus: dict[str, dict[str, float]] = {
        "CRAWLED_ORIG": compute_avg(scores_crawled_orig),
        "DEPLAIN_ORIG": compute_avg(scores_deplain_orig),
        "G4A_ORIG": compute_avg(scores_g4a_orig),
        "CRAWLED_PLAIN": compute_avg(scores_crawled_plain),
        "DEPLAIN_PLAIN": compute_avg(scores_deplain_plain),
        "G4A_PLAIN": compute_avg(scores_g4a_plain),
        "CRAWLED_EASY": compute_avg(scores_crawled_easy),
        "Leiko_PLAIN": compute_avg(scores_leiko_plain),
    }

    # Plot violin + swarm for sentence length distribution across corpora.
    _plot_sentence_lengths(
        {
            "CRAWLED_ORIG": scores_crawled_orig.get(feature_names[0], []),
            "CRAWLED_PLAIN": scores_crawled_plain.get(feature_names[0], []),
            "CRAWLED_EASY": scores_crawled_easy.get(feature_names[0], []),
            "DEPLAIN_ORIG": scores_deplain_orig.get(feature_names[0], []),
            "DEPLAIN_PLAIN": scores_deplain_plain.get(feature_names[0], []),
            "G4A_ORIG": scores_g4a_orig.get(feature_names[0], []),
            "G4A_PLAIN": scores_g4a_plain.get(feature_names[0], []),
            "Leiko_PLAIN": scores_leiko_plain.get(feature_names[0], []),
        },
        "Sentence Length Distribution (Tokens per Sentence)",
        Path("output") / "sentence_length.png",
    )

    # ---------------------------------------------------------------------
    # Collect all sentence lengths together with corpus and document name,
    # then plot the top N longest sentences (default 20).
    # ---------------------------------------------------------------------
    def _collect_lengths(corpus_path: Path, label: str) -> list[tuple[float, str]]:
        """Return a list of (length, identifier) for every sentence in the corpus.

        ``identifier`` combines the corpus label and the file name for easy
        reference in the plot (e.g., "CRAWLED_ORIG/file1.txt").
        """
        cache_dir = Path(__file__).parent / "cache" / utils_mod.get_corpus_slug(corpus_path)
        cache_dir.mkdir(parents=True, exist_ok=True)
        entries: list[tuple[float, str]] = []
        for file in corpus_path.iterdir():
            if not file.is_file():
                continue
            result = utils_mod._process_file(
                file,
                use_cache=True,
                cache_dir=cache_dir,
                feature_extractor=extractor,
                feature_names=feature_names,
            )
            lengths = result.get(feature_names[0], [])
            for length in lengths:
                identifier = f"{label}/{file.name}"
                entries.append((float(length), identifier))
        return entries

    all_entries: list[tuple[float, str]] = []
    for label, path in {
        "CRAWLED_ORIG": CRAWLED_ORIG,
        "CRAWLED_PLAIN": CRAWLED_PLAIN,
        "CRAWLED_EASY": CRAWLED_EASY,
        "G4A_ORIG": GFA_ORIG,
        "G4A_PLAIN": GFA_PLAIN,
        "DEPLAIN_ORIG": DEPLAIN_ORIG,
        "DEPLAIN_PLAIN": DEPLAIN_PLAIN,
        "Leiko_PLAIN": Leiko_PLAIN,
    }.items():
        all_entries.extend(_collect_lengths(path, label))

    # Sort by length descending and keep top 20.
    top_n = 20
    top_entries = sorted(all_entries, key=lambda x: x[0], reverse=True)[:top_n]

    # Print the top lengths to the console.
    print(f"Top {top_n} longest sentences across all corpora (tokens per sentence):")
    for length, identifier in top_entries:
        print(f"{identifier}: {length:.2f}")

if __name__ == "__main__":
    main()
