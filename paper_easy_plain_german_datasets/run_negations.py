import re
import pandas as pd
import os
from pathlib import Path
from statistics import mean
from cassis import Cas
from tqdm import tqdm

import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from typing import Dict,List,Mapping
from utils import get_corpus_slug, _category
from constants import (
    TIGER_ORIG,
    ASGC_ORIG,
    ASGC_PLAIN,
    ASGC_EASY,
    G4A_ORIG,
    G4A_EASY,
    DEplain_ORIG,
    DEplain_PLAIN,
    Leiko_EASY
)
from py_lift.decorators import supported_languages
from py_lift.dkpro import T_LEMMA
from py_lift.annotators.api import SEL_BaseAnnotator
from py_lift.annotators.lists import SEL_ListReader
from typing import Optional
from py_lift.utils.core import load_cas_from_xmi_with_lift_ts  # type: ignore
from py_lift.preprocessing import Spacy_Preprocessor  # type: ignore
from py_lift.utils.core import load_lift_typesystem  # type: ignore

# Initialise the preprocessing pipeline once – it is safe to reuse across calls.
prep = Spacy_Preprocessor("de", auto_install_models=True)
ts = load_lift_typesystem()

# ad hoc type
NEG = ts.create_type(name='org.lift.type.Negation')

@supported_languages('de')
class SE_NegationAnnotator(SEL_BaseAnnotator, SEL_ListReader):

    def __init__(self, language, ts=None):
        filename = Path(__file__).parent / "negation_words.txt"
        SEL_ListReader.__init__(self, filename)
        SEL_BaseAnnotator.__init__(self, language)

    def _process(self, cas: Cas) -> bool:
        negations = self.read_list()
        for lemma in cas.select(T_LEMMA):
            l_str = lemma.value
            if l_str in negations:
                neg_anno = NEG(begin=lemma.get('begin'), end=lemma.get('end'))
                cas.add(neg_anno)
        return True

def run_folder(
    dir: Path,
    use_cache: bool = True,
) -> float:

    if use_cache:
        cache_dir = Path(__file__).parent / "cache" / get_corpus_slug(dir)
        cache_dir.mkdir(parents=True, exist_ok=True)

    file_count = 0
    docs_with_negation = 0

    for file in tqdm(dir.iterdir(), desc=f"Processing {dir.name}"):
        if not file.is_file():
            continue

        cas = get_cas_for_file(file, use_cache, cache_dir)

        file_count += 1

        # If any token in the document is negation, count the document.
        if has_negation(cas):
            docs_with_negation += 1

    # Proportion of documents that contain at least one negation.
    return docs_with_negation / file_count if file_count > 0 else float("nan")


    return aggregated

def get_cas_for_file(
    file: Path,
    use_cache: bool,
    cache_dir: Path | None,
) -> Cas:

    try:
        if use_cache and cache_dir is not None:
            cache_file = cache_dir / f"{file.stem}.xmi"
            # Load from cache if present, otherwise process and cache.
            if cache_file.is_file():
                cas = load_cas_from_xmi_with_lift_ts(cache_file)
            else:
                text = file.read_text(encoding="utf-8")
                cas = prep.run(text)
                # Cache the XMI for future runs.
                cas.to_xmi(str(cache_file))
        else:
            # Direct processing without caching.
            text = file.read_text(encoding="utf-8")
            cas = prep.run(text)

        return cas
    except Exception as e:
        raise RuntimeError(f"Error processing file '{file}': {e}") from e

def has_negation(cas: Cas) -> bool:
    negation_annotator = SE_NegationAnnotator(language="de", ts=ts)
    negation_annotator.process(cas)
    return any(cas.select("org.lift.type.Negation"))

def plot_negation_proportions(results: Dict[str, float]) -> None:
    """Create a horizontal bar chart using pre‑computed *results*.

    ``results`` should map a corpus identifier (e.g. ``"TIGER_ORIG"``) to the
    negation proportion that was already calculated in ``main``. This avoids
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
    plt.xlabel("Negation proportion")
    plt.title("Negation proportion per corpus")
    ax = plt.gca()
    plt.setp(ax.get_yticklabels(), rotation=0, ha="right")

    # Adjust margins to match other plots.
    fig = plt.gcf()
    fig.subplots_adjust(left=0.30, right=0.95)
    plt.tight_layout()
    
    output_path = Path("output") / "negation_proportions.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

def main() -> None:

    score_tiger_orig = run_folder(TIGER_ORIG)

    score_asgc_orig = run_folder(ASGC_ORIG)
    score_asgc_plain = run_folder(ASGC_PLAIN)
    score_asgc_easy = run_folder(ASGC_EASY)

    score_g4a_orig = run_folder(G4A_ORIG)
    score_g4a_easy = run_folder(G4A_EASY)

    score_deplain_orig = run_folder(DEplain_ORIG)
    score_deplain_plain = run_folder(DEplain_PLAIN)

    score_leiko_easy = run_folder(Leiko_EASY)

    plot_negation_proportions(
        {
            "TIGER_ORIG": score_tiger_orig,
            "ASGC_ORIG": score_asgc_orig,
            "DEplain_ORIG": score_deplain_orig,
            "G4A_ORIG": score_g4a_orig,
            "ASGC_PLAIN": score_asgc_plain,
            "DEplain_PLAIN": score_deplain_plain,
            "ASGC_EASY": score_asgc_easy,
            "G4A_EASY": score_g4a_easy,
            "Leiko_EASY": score_leiko_easy,
        }
    )


if __name__ == "__main__":
    main()