"""Utility functions for computing readability scores.

Both the command‑line script ``run_simple.py`` and the Jupyter notebook share
the same logic for loading text files, optionally using a cached XMI representation,
and extracting the ``Readability_Score_FleschReadingEase_de`` feature.

The functions below encapsulate that behaviour so it can be reused without code
duplication.
"""

import os
from pathlib import Path
from tqdm import tqdm
from py_lift.preprocessing import Spacy_Preprocessor
from py_lift.utils.core import load_lift_typesystem
from py_lift.readability import FE_TextstatFleschIndex
from py_lift.utils.core import load_cas_from_xmi_with_lift_ts

# Initialise the preprocessing pipeline once – it is safe to reuse across calls.
prep = Spacy_Preprocessor("de", auto_install_models=True)
ts = load_lift_typesystem()


def get_corpus_slug(folder: Path) -> str:
    """Return a normalized slug for a corpus folder.

    The slug is derived from the full folder path with path separators replaced
    by hyphens. This ensures a unique cache directory name for each corpus.
    """
    return str(folder).replace(os.sep, "-")

def _extract_scores_from_cas(cas) -> list[float]:
    """Extract the Flesch readability score from a CAS.

    Returns a list with a single float score if the feature is present, otherwise an empty list.
    """
    FE_TextstatFleschIndex("de").extract(cas)
    for feature in cas.select("org.lift.type.FeatureAnnotationNumeric"):
        if feature.get('name') == 'Readability_Score_FleschReadingEase_de':
            try:
                return [float(feature.value)]
            except Exception as e:
                raise RuntimeError(
                    f"Failed to convert feature value {feature.value!r}: {e}"
                )
    return []


def _load_or_process(file: Path, cache_file: Path) -> list[float]:
    """Load a cached CAS or process the raw text and cache the result.

    Returns a list containing the single readability score extracted from the
    CAS. The caller aggregates these values.
    """
    if cache_file.is_file():
        cas = load_cas_from_xmi_with_lift_ts(cache_file)
    else:
        text = file.read_text(encoding="utf-8")
        cas = prep.run(text)
        # Cache the XMI for future runs.
        cas.to_xmi(str(cache_file))
    return _extract_scores_from_cas(cas)


def process_folder(dir: Path, use_cache: bool = True) -> list[float]:
    """Process *dir* and return all readability scores.

    Parameters
    ----------
    dir: Path
        Directory containing the raw text files.
    use_cache: bool, default ``True``
        When ``True`` the function will look for a cached ``.xmi`` file for each
        document and store newly processed CAS objects. When ``False`` the cache
        is bypassed and the raw text is processed on‑the‑fly. This is useful for
        interactive notebook sessions where you want the most up‑to‑date results
        without side‑effects on the filesystem.
    """
    if use_cache:
        cache_dir = Path(__file__).parent / "cache" / get_corpus_slug(dir)
        cache_dir.mkdir(parents=True, exist_ok=True)

    scores: list[float] = []
    for file in tqdm(dir.iterdir(), desc=f"Processing {dir.name}"):
        if not file.is_file():
            continue
        if use_cache:
            cache_file = cache_dir / f"{file.stem}.xmi"
            scores.extend(_load_or_process(file, cache_file))
        else:
            # Direct processing without caching.
            text = file.read_text(encoding="utf-8")
            cas = prep.run(text)
            scores.extend(_extract_scores_from_cas(cas))
    return scores
