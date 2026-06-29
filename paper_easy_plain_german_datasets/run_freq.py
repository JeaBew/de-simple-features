import pandas as pd
import os
from pathlib import Path
from statistics import mean

import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from typing import Optional
from py_lift.utils.core import load_cas_from_xmi_with_lift_ts
from typing import Mapping

def plot_stacked_bars(avg_scores_by_corpus, output_path):
    first_corpus_scores = next(iter(avg_scores_by_corpus.values()))
    feature_order = list(first_corpus_scores.keys())

    corpus_labels = list(avg_scores_by_corpus.keys())

    # ↓ Make overall chart less wide (reduce first number)
    fig, ax = plt.subplots(figsize=(6, 6))  # e.g. (6,6) instead of (8,6)

    colors = sns.color_palette("colorblind", len(feature_order))
    cumulative = {label: 0.0 for label in corpus_labels}

    for idx, feature in enumerate(feature_order):
        heights, bottoms = [], []
        for label in corpus_labels:
            avg = avg_scores_by_corpus[label].get(feature, np.nan)
            height = 0.0 if np.isnan(avg) else float(avg)
            heights.append(height)
            bottoms.append(cumulative[label])

        ax.bar(
            corpus_labels,
            heights,
            bottom=bottoms,
            label=feature,
            color=colors[idx % len(colors)],
            width=0.35,
        )

        for label, h in zip(corpus_labels, heights):
            cumulative[label] += h

    ax.set_title("Average Frequency Ratio Features by Corpus")
    # Hide y‑axis ticks and labels for a cleaner appearance.
    ax.set_yticks([])
    ax.set_yticklabels([])
    plt.setp(ax.get_xticklabels(), rotation=90, ha="center")
    fig.subplots_adjust(bottom=0.35)
    
    # ↓ Combine custom legend names with a legend placed outside the axes.
    # First retrieve the automatically generated handles (the colored patches).
    handles, _ = ax.get_legend_handles_labels()
    # Map the original feature names to the desired short names.
    feature_map = {
        "Freq_Ratio_F7_PER_Token": "F7",
        "Freq_Ratio_F6_PER_Token": "F6",
        "Freq_Ratio_F5_PER_Token": "F5",
        "Freq_Ratio_F4_PER_Token": "F4",
        "Freq_Ratio_F3_PER_Token": "F3",
        "Freq_Ratio_F2_PER_Token": "F2",
        "Freq_Ratio_F1_PER_Token": "F1",
        "Freq_Ratio_OOV_PER_Token": "OOV",
    }
    custom_labels = [feature_map.get(f, f) for f in feature_order]
    
    # ---- Invert the order ----
    handles = handles[::-1]          # reverse the list of patches
    custom_labels = custom_labels[::-1]  # reverse the list of label strings

    # Use the figure's legend method so we can position it outside the axes.
    fig.legend(
        handles,
        custom_labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.97),
        ncol=4,  # display legend entries in three columns
        frameon=False,
    )
        
    # Reserve room at top for the legend
    fig.subplots_adjust(top=0.80)  # smaller => more space for legend

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight", pad_inches=0.2)
    plt.show()

def plot_grouped_bars(avg_scores_by_corpus, output_path):
    """Create a grouped bar chart where *features* are on the x‑axis.

    The x‑labels are the frequency‑ratio features (``F7`` … ``F1`` and ``OOV``).
    For each feature a set of bars – one per corpus – shows the average value
    for that corpus.  This is the inverse of the previous grouping where the
    corpora were on the x‑axis.
    """
    # Determine ordering based on the first corpus (consistent with the stacked chart).
    first_corpus_scores = next(iter(avg_scores_by_corpus.values()))
    feature_order = list(first_corpus_scores.keys())
    corpus_labels = list(avg_scores_by_corpus.keys())

    # Figure size similar to the stacked version.
    fig, ax = plt.subplots(figsize=(6, 6))

    colors = sns.color_palette("colorblind", len(corpus_labels))
    n_corpora = len(corpus_labels)
    total_width = 0.8
    bar_width = total_width / n_corpora
    # Base positions for each feature group.
    indices = np.arange(len(feature_order))

    for idx, corpus in enumerate(corpus_labels):
        heights = []
        for feature in feature_order:
            avg = avg_scores_by_corpus[corpus].get(feature, np.nan)
            height = 0.0 if np.isnan(avg) else float(avg)
            heights.append(height)
        # Compute offset so bars are centered on the tick.
        offset = (idx - n_corpora / 2) * bar_width + bar_width / 2
        ax.bar(
            indices + offset,
            heights,
            width=bar_width,
            label=corpus,
            color=colors[idx % len(colors)],
        )

    ax.set_title("Average Frequency Ratio Features by Corpus (Grouped)")
    # Map feature keys to short names for the x‑axis.
    feature_map = {
        "Freq_Ratio_F7_PER_Token": "F7",
        "Freq_Ratio_F6_PER_Token": "F6",
        "Freq_Ratio_F5_PER_Token": "F5",
        "Freq_Ratio_F4_PER_Token": "F4",
        "Freq_Ratio_F3_PER_Token": "F3",
        "Freq_Ratio_F2_PER_Token": "F2",
        "Freq_Ratio_F1_PER_Token": "F1",
        "Freq_Ratio_OOV_PER_Token": "OOV",
    }
    short_features = [feature_map.get(f, f) for f in feature_order]
    ax.set_xticks(indices)
    ax.set_xticklabels(short_features, rotation=0, ha="center")
    ax.set_ylabel("Average Ratio")
    # Hide y‑axis ticks for a cleaner look, similar to stacked version.
    ax.set_yticks([])
    ax.set_yticklabels([])

    # Legend now shows corpus names.
    handles, _ = ax.get_legend_handles_labels()
    # Invert order so legend matches the bar order (optional, keep consistent with stacked).
    handles = handles[::-1]
    corpus_labels_rev = corpus_labels[::-1]
    fig.legend(
        handles,
        corpus_labels_rev,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.97),
        ncol=4,
        frameon=False,
    )
    fig.subplots_adjust(top=0.80, bottom=0.35)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight", pad_inches=0.2)
    plt.show()

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
from py_lift.annotators.frequency import SE_TokenZipfFrequency
from py_lift.extractors_specific import FE_FreqBandRatios

BAND_ORDER = ["f7", "f6", "f5", "f4", "f3", "f2", "f1", "oov"]


def get_freq_token(
    corpus_dir: Path,
    out_csv: Optional[Path] = None,
    lang: str = "de",
) -> pd.DataFrame:
    """
    Aggregiert pro Wort (lowercased) über alle XMIs in corpus_dir:
    - count: Häufigkeit (Tokenanzahl)
    - band: Frequency band (Mehrheitsentscheidung, falls inkonsistent)

    Erwartet Frequency-Annotationen vom Typ 'org.lift.type.Frequency' mit Feature 'frequencyBand'.
    """
    # word -> total token count
    word_counts: dict[str, int] = {}
    # word -> band -> count (um Mehrheitsband zu wählen)
    word_band_votes: dict[str, dict[str, int]] = {}

    xmi_files = sorted(corpus_dir.glob("*.xmi"))
    for xmi_file in tqdm(xmi_files, desc=f"Processing {corpus_dir}"):
        cas = load_cas_from_xmi_with_lift_ts(xmi_file)

        # Falls Frequency-Annos noch nicht drin sind, annotieren:
        SE_TokenZipfFrequency(lang).process(cas)

        # Map: (begin,end) -> frequencyBand
        freq_by_span: dict[tuple[int, int], str] = {}
        for freq in cas.select("org.lift.type.Frequency"):
            band = freq.get("frequencyBand")
            if band is None:
                continue
            freq_by_span[(int(freq.begin), int(freq.end))] = str(band).strip().lower()

        for tok in cas.select("Token"):
            word = tok.get_covered_text()
            if not word:
                continue
            word = word.strip().lower()
            if not word:
                continue

            span = (int(tok.begin), int(tok.end))
            band = freq_by_span.get(span, "oov")  # fallback, falls keine Frequency gefunden

            word_counts[word] = word_counts.get(word, 0) + 1
            word_band_votes.setdefault(word, {})
            word_band_votes[word][band] = word_band_votes[word].get(band, 0) + 1

    # Ergebnis-Tabelle bauen
    rows = []
    for word, cnt in word_counts.items():
        votes = word_band_votes.get(word, {})
        # wähle Band mit meisten Votes; Tie-break nach BAND_ORDER
        def rank(b: str) -> int:
            return BAND_ORDER.index(b) if b in BAND_ORDER else 999

        best_band = sorted(votes.items(), key=lambda kv: (-kv[1], rank(kv[0])))[0][0] if votes else "oov"
        rows.append({"word": word, "count": cnt, "band": best_band})

    df = pd.DataFrame(rows)
    df["band"] = pd.Categorical(df["band"], categories=BAND_ORDER, ordered=True)
    df = df.sort_values(by=["band", "count", "word"], ascending=[True, False, True]).reset_index(drop=True)

    if out_csv is not None:
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out_csv, index=False)

    return df





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

    scores_tiger_orig = process_folder(TIGER_ORIG, feature_extractor=extractor, feature_names=feature_names)

    scores_asgc_orig = process_folder(ASGC_ORIG, feature_extractor=extractor, feature_names=feature_names)
    scores_asgc_plain = process_folder(ASGC_PLAIN, feature_extractor=extractor, feature_names=feature_names)
    scores_asgc_easy = process_folder(ASGC_EASY, feature_extractor=extractor, feature_names=feature_names)

    scores_g4a_orig = process_folder(G4A_ORIG, feature_extractor=extractor, feature_names=feature_names)
    scores_g4a_plain = process_folder(G4A_PLAIN, feature_extractor=extractor, feature_names=feature_names)

    scores_deplain_orig = process_folder(DEplain_ORIG, feature_extractor=extractor, feature_names=feature_names)
    scores_deplain_plain = process_folder(DEplain_PLAIN, feature_extractor=extractor, feature_names=feature_names)

    scores_leiko_plain = process_folder(Leiko_PLAIN, feature_extractor=extractor, feature_names=feature_names)
    
    # ---------------------------------------------------------------------
    # Compute the average score for each feature across all documents.
    # ---------------------------------------------------------------------
    # Compute average scores per feature for each corpus.
    def compute_avg(scores_dict: dict[str, list[float]]) -> dict[str, float]:
        return {name: (mean(scores_dict.get(name, [])) if scores_dict.get(name) else float('nan')) for name in feature_names}

    avg_by_corpus: dict[str, dict[str, float]] = {
        "TIGER_ORIG": compute_avg(scores_tiger_orig),
        "ASGC_ORIG": compute_avg(scores_asgc_orig),
        "ASGC_PLAIN": compute_avg(scores_asgc_plain),
        "ASGC_EASY": compute_avg(scores_asgc_easy),
        "G4A_ORIG": compute_avg(scores_g4a_orig),
        "G4A_PLAIN": compute_avg(scores_g4a_plain),
        "DEplain_ORIG": compute_avg(scores_deplain_orig),
        "DEplain_PLAIN": compute_avg(scores_deplain_plain),
        "Leiko_PLAIN": compute_avg(scores_leiko_plain),
    }

    # Plot stacked bars for all corpora.
    plot_stacked_bars(avg_by_corpus, output_path=Path("output") / "freq_stacked.png")
    # Plot grouped bars (features on x‑axis) for all corpora.
    plot_grouped_bars(avg_by_corpus, output_path=Path("output") / "freq_grouped.png")
    plot_grouped_bars(avg_by_corpus, output_path=Path("output") / "freq_grouped.png")




if __name__ == "__main__":
    main()
