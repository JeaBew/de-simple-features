import pandas as pd
import os
from pathlib import Path
from statistics import mean

import seaborn as sns
import matplotlib.pyplot as plt

from utils import process_folder
from constants import CRAWLED_ORIG, CRAWLED_PLAIN, CRAWLED_EASY, GFA_ORIG, GFA_PLAIN, DEPLAIN_ORIG, DEPLAIN_PLAIN
from py_lift.readability import FE_TextstatWienerSachtextformel_1  # type: ignore

def _plot_scores(corpus_scores: dict, title: str, output_path: Path) -> None:
    """Create a violin + swarm plot for the given corpus scores.

    Args:
        corpus_scores: Mapping from corpus name (e.g., "orig", "plain", "easy")
            to a list of readability scores.
        title: Title for the plot.
        output_path: File path where the PNG image will be saved.
    """
    # Prepare DataFrame
    df = pd.DataFrame({
        "corpus": sum([[name] * len(scores) for name, scores in corpus_scores.items()], []),
        "score": sum([list(scores) for scores in corpus_scores.values()], []),
    })

    sns.set(style="whitegrid")
    plt.figure(figsize=(8, 6))
    # Violin plot with hue to avoid deprecation warning
    sns.violinplot(x="corpus", y="score", hue="corpus", data=df, inner="quartile", palette="muted", legend=False)
    # Swarm overlay
    sns.swarmplot(x="corpus", y="score", data=df, color="k", alpha=0.5, size=3)
    plt.title(title)
    plt.ylabel("Readability Score")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300)
    plt.close()


def _plot_box(corpus_scores: dict, title: str, output_path: Path) -> None:
    """Create a box plot (separate from the violin plot) for the given scores.

    Args:
        corpus_scores: Mapping from corpus name to list of scores.
        title: Plot title.
        output_path: Destination PNG file.
    """
    df = pd.DataFrame({
        "corpus": sum([[name] * len(scores) for name, scores in corpus_scores.items()], []),
        "score": sum([list(scores) for scores in corpus_scores.values()], []),
    })

    sns.set(style="whitegrid")
    plt.figure(figsize=(8, 6))
    sns.boxplot(x="corpus", y="score", data=df, palette="muted", hue="corpus")
    plt.title(title)
    plt.ylabel("Readability Score")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300)
    plt.close()


def main() -> None:
    # The extractor now receives the CAS directly. We wrap the original class
    # method to match the new ``feature_extractor`` signature.
    extractor = lambda cas: FE_TextstatWienerSachtextformel_1().extract(cas)
    feature_names = ["Readability_Score_WienerSachtextformel-1_de"]
    
    # Gather all readability scores from the crawled corpora.
    scores_crawled_orig = process_folder(CRAWLED_ORIG, feature_extractor=extractor, feature_names=feature_names)
    scores_crawled_plain = process_folder(CRAWLED_PLAIN, feature_extractor=extractor, feature_names=feature_names)
    scores_crawled_easy = process_folder(CRAWLED_EASY, feature_extractor=extractor, feature_names=feature_names)

    # Compute and output the mean readability score for each crawled corpus.
    # Extract the list for the single feature we are interested in.
    feat = feature_names[0]
    mean_orig = mean(scores_crawled_orig.get(feat, [])) if scores_crawled_orig else float('nan')
    mean_plain = mean(scores_crawled_plain.get(feat, [])) if scores_crawled_plain else float('nan')
    mean_easy = mean(scores_crawled_easy.get(feat, [])) if scores_crawled_easy else float('nan')
    print(f"Mean readability (crawled orig): {mean_orig}")
    print(f"Mean readability (crawled plain): {mean_plain}")
    print(f"Mean readability (crawled easy): {mean_easy}")

    # Plot for crawled dataset
    _plot_scores(
        {"orig": scores_crawled_orig.get(feat, []), "plain": scores_crawled_plain.get(feat, []), "easy": scores_crawled_easy.get(feat, [])},
        "Crawled Corpus Readability Distribution",
        Path("output") / "crawled_readability.png",
    )
    # Additional box plot for crawled dataset
    _plot_box(
        {"orig": scores_crawled_orig.get(feat, []), "plain": scores_crawled_plain.get(feat, []), "easy": scores_crawled_easy.get(feat, [])},
        "Crawled Corpus Readability Box Plot",
        Path("output") / "crawled_box.png",
    )

    # German4All dataset
    scores_g4a_orig = process_folder(GFA_ORIG, feature_extractor=extractor, feature_names=feature_names)
    scores_g4a_plain = process_folder(GFA_PLAIN, feature_extractor=extractor, feature_names=feature_names)
    mean_g4a_orig = mean(scores_g4a_orig.get(feat, [])) if scores_g4a_orig else float('nan')
    mean_g4a_plain = mean(scores_g4a_plain.get(feat, [])) if scores_g4a_plain else float('nan')
    print(f"Mean readability (g4a orig): {mean_g4a_orig}")
    print(f"Mean readability (g4a plain): {mean_g4a_plain}")
    _plot_scores(
        {"orig": scores_g4a_orig.get(feat, []), "plain": scores_g4a_plain.get(feat, [])},
        "German4All Readability Distribution",
        Path("output") / "g4a_readability.png",
    )
    # Additional box plot for German4All dataset
    _plot_box(
        {"orig": scores_g4a_orig.get(feat, []), "plain": scores_g4a_plain.get(feat, [])},
        "German4All Readability Box Plot",
        Path("output") / "g4a_box.png",
    )

    # Deplain dataset
    scores_deplain_orig = process_folder(DEPLAIN_ORIG, feature_extractor=extractor, feature_names=feature_names)
    scores_deplain_plain = process_folder(DEPLAIN_PLAIN, feature_extractor=extractor, feature_names=feature_names)
    mean_deplain_orig = mean(scores_deplain_orig.get(feat, [])) if scores_deplain_orig else float('nan')
    mean_deplain_plain = mean(scores_deplain_plain.get(feat, [])) if scores_deplain_plain else float('nan')
    print(f"Mean readability (deplain orig): {mean_deplain_orig}")
    print(f"Mean readability (deplain plain): {mean_deplain_plain}")
    _plot_scores(
        {"orig": scores_deplain_orig.get(feat, []), "plain": scores_deplain_plain.get(feat, [])},
        "Deplain Readability Distribution",
        Path("output") / "deplain_readability.png",
    )
    # Additional box plot for Deplain dataset
    _plot_box(
        {"orig": scores_deplain_orig.get(feat, []), "plain": scores_deplain_plain.get(feat, [])},
        "Deplain Readability Box Plot",
        Path("output") / "deplain_box.png",
    )

if __name__ == "__main__":
    main()
