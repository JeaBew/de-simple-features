import pandas as pd
import os
from pathlib import Path
from statistics import mean

import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from typing import Mapping

def plot_stacked_bars(
    avg_scores_by_corpus: Mapping[str, Mapping[str, float]],
    output_path: Path,
) -> None:
    """Plot stacked bars for multiple corpora.

    Parameters
    ----------
    avg_scores_by_corpus: Mapping[str, Mapping[str, float]]
        Mapping from corpus label to a mapping of feature name → average score.
    output_path: Path
        Fully‑qualified path where the PNG image will be saved.
    """
    # Determine a consistent feature order using the first corpus.
    first_corpus_scores = next(iter(avg_scores_by_corpus.values()))
    feature_order = list(first_corpus_scores.keys())

    corpus_labels = list(avg_scores_by_corpus.keys())
    fig, ax = plt.subplots(figsize=(10, 6))
    # Use a colorblind‑friendly qualitative palette to give each feature a distinct
    # and easily readable color.
    # seaborn provides a "colorblind" palette that works well for stacked bar charts.
    colors = sns.color_palette("colorblind", len(feature_order))

    # Track cumulative heights for each corpus.
    cumulative: dict[str, float] = {label: 0.0 for label in corpus_labels}

    for idx, feature in enumerate(feature_order):
        heights = []
        bottoms = []
        for label in corpus_labels:
            avg = avg_scores_by_corpus[label].get(feature, float('nan'))
            height = avg if not isinstance(avg, float) or not avg != avg else 0
            heights.append(height)
            bottoms.append(cumulative[label])
        ax.bar(
            corpus_labels,
            heights,
            bottom=bottoms,
            label=feature,
            color=colors[idx % len(colors)],
        )
        # Update cumulative bottoms.
        for label, h in zip(corpus_labels, heights):
            cumulative[label] += h

    ax.set_ylabel("Average Score")
    ax.set_title("Average Frequency Ratio Features by Corpus")
    ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    # Rotate corpus labels for better readability.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    plt.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300)
    plt.show()

from utils import process_folder
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
from py_lift.annotators.frequency import SE_TokenZipfFrequency
from py_lift.extractors_specific import FE_FreqBandRatios

def main() -> None:
    extractor = lambda cas: (
        SE_TokenZipfFrequency("de").process(cas),
        FE_FreqBandRatios().extract(cas),
    )
    feature_names = [
            'Freq_Ratio_OOV_PER_Token',
            'Freq_Ratio_F1_PER_Token',
            'Freq_Ratio_F2_PER_Token',
            'Freq_Ratio_F3_PER_Token',
            'Freq_Ratio_F4_PER_Token',
            'Freq_Ratio_F5_PER_Token',
            'Freq_Ratio_F6_PER_Token',
            'Freq_Ratio_F7_PER_Token'
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

    # Plot stacked bars for all corpora.
    plot_stacked_bars(avg_by_corpus, output_path=Path("output") / "freq_stacked.png")

if __name__ == "__main__":
    main()
