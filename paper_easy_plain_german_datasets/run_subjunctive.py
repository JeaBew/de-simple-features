import seaborn as sns
import spacy
from tqdm import tqdm
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional
import pandas as pd
from utils import process_folder, _category, get_corpus_slug, _process_file
from constants import (
    TIGER_ORIG,
    ASGC_ORIG,
    ASGC_PLAIN,
    ASGC_EASY,
    G4A_ORIG,
    G4A_EASY,
    DEplain_ORIG,
    DEplain_PLAIN,
    Leiko_EASY,
)

# de_core_news_sm/md/lg oder de_dep_news_trf etc.
nlp = spacy.load("de_core_news_lg")

def is_subjunctive(tok):
    return classify_subjunctive(tok) is not None

def classify_subjunctive(tok):
    """
    Returns: "KI", "KII", "CND", or None
    Rule of thumb:
      - KI  ~ Mood=Sub & Tense=Pres
      - KII ~ Mood=Sub & Tense=Past
      - würde-Konstruktion oft Mood=Cnd (optional)
    """
    if tok.pos_ not in ("AUX", "VERB"):
        return None
    if tok.morph.get("VerbForm") and "Fin" not in tok.morph.get("VerbForm"):
        return None

    mood = tok.morph.get("Mood")
    tense = tok.morph.get("Tense")

    if "Sub" in mood:
        if "Pres" in tense:
            return "KI"
        if "Past" in tense:
            return "KII"
        # Konjunktiv, aber Tense nicht vorhanden/unklar
        return "KONJ_SUB_UNKLAR"

    # Optional: würde-Formen als Konditional markieren
    if "Cnd" in mood:
        return "CND"

    return None


def process_folder(
    dir: Path,
) -> float:
    
    # Count how many documents contain at least one subjunctive token.
    file_count = 0
    docs_with_subjunctive = 0
    files = list(dir.iterdir())
    for file in tqdm(files, desc=f"Processing {dir.name}"):
        if not file.is_file():
            continue

        text = file.read_text(encoding="utf-8")
        doc = nlp(text)
        file_count += 1

        # If any token in the document is subjunctive, count the document.
        if any(is_subjunctive(tok) for tok in doc):
            docs_with_subjunctive += 1

    # Proportion of documents that contain at least one subjunctive form.
    return docs_with_subjunctive / file_count if file_count > 0 else float("nan")

def plot_subjunctive_proportions(results: Dict[str, float]) -> None:
    """Create a horizontal bar chart using pre‑computed *results*.

    ``results`` should map a corpus identifier (e.g. ``"TIGER_ORIG"``) to the
    subjunctive proportion that was already calculated in ``main``. This avoids
    re‑scanning the directories.
    """
    import matplotlib.pyplot as plt

    # Build a DataFrame that also encodes a categorical colour mapping.
    # Use the explicit sum pattern to preserve the insertion order of the
    # ``results`` dictionary (matching the style used in other plotting scripts).
    df = pd.DataFrame({
        "corpus": sum([[name] for name in results.keys()], []),
        "proportion": sum([[value] for value in results.values()], []),
    })

    df["category"] = df["corpus"].apply(_category)

    sns.set(style="whitegrid")
    plt.figure(figsize=(6, max(2, len(df) * 0.4)))

    palette = {"orig": "tab:blue", "plain": "tab:orange", "easy": "tab:green"}
    sns.barplot(
        x="proportion",
        y="corpus",
        hue="category",
        data=df,
        palette=palette,
        dodge=False,
        legend=False,
    )
    plt.xlabel("Subjunctive proportion")
    plt.title("Subjunctive proportion per corpus")
    ax = plt.gca()
    plt.setp(ax.get_yticklabels(), rotation=0, ha="right")

    # Adjust margins to match other plots.
    fig = plt.gcf()
    fig.subplots_adjust(left=0.30, right=0.95)
    plt.tight_layout()
    
    output_path = Path("output") / "subjunctive_proportions.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

if __name__ == "__main__":
    # Compute proportions for each individual corpus.
    tiger_results = process_folder(TIGER_ORIG)
    asgc_orig_results = process_folder(ASGC_ORIG)
    asgc_plain_results = process_folder(ASGC_PLAIN)
    asgc_easy_results = process_folder(ASGC_EASY)
    g4a_orig_results = process_folder(G4A_ORIG)
    g4a_easy_results = process_folder(G4A_EASY)
    deplain_orig_results = process_folder(DEplain_ORIG)
    deplain_plain_results = process_folder(DEplain_PLAIN)
    leiko_easy_results = process_folder(Leiko_EASY)
    print(f"Leiko_EASY subjunctive proportion: {leiko_easy_results:.4f}")

    # Gather results into a single dict and plot using the pre‑computed values.
    results = {
        "TIGER_ORIG": tiger_results,
        "ASGC_ORIG": asgc_orig_results,
        "DEplain_ORIG": deplain_orig_results,
        "G4A_ORIG": g4a_orig_results,
        "ASGC_PLAIN": asgc_plain_results,
        "DEplain_PLAIN": deplain_plain_results,
        "ASGC_EASY": asgc_easy_results,
        "G4A_EASY": g4a_easy_results,
        "Leiko_EASY": leiko_easy_results,
    }
    plot_subjunctive_proportions(results)
    