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
from typing import Callable, Any, List, Dict
# The ``py_lift`` library provides the linguistic preprocessing and feature
# extraction utilities used by this project. It may not be available in every
# development environment, so we silence static‑analysis errors.
from py_lift.preprocessing import Spacy_Preprocessor  # type: ignore
from py_lift.utils.core import load_lift_typesystem  # type: ignore
from py_lift.utils.core import load_cas_from_xmi_with_lift_ts  # type: ignore

# Initialise the preprocessing pipeline once – it is safe to reuse across calls.
prep = Spacy_Preprocessor("de", auto_install_models=True)
ts = load_lift_typesystem()

# ---------------------------------------------------------------------------
# Helper to combine several ``py_lift`` feature extractors into a single callable.
# ---------------------------------------------------------------------------
def chain_extractors(*extractors: Any) -> Callable[[Any], None]:
    """Return a callable that runs *extractors* sequentially on the same CAS.

    Each *extractor* is expected to be an *instance* (or a class that can be
    instantiated without arguments) providing an ``extract(cas)`` method – the
    same contract used by the original ``FE_TextstatWienerSachtextformel_1``.

    The returned function matches the ``feature_extractor`` signature accepted by
    :func:`process_folder` and :func:`_process_file` – it receives a CAS object
    and invokes ``extract`` on each supplied extractor in order.

    Example
    -------
    >>> from py_lift.readability import FE_TextstatWienerSachtextformel_1, FE_OtherFeature
    >>> chained = chain_extractors(FE_TextstatWienerSachtextformel_1(), FE_OtherFeature())
    >>> scores = process_folder(Path("data"), feature_extractor=chained,
    ...                       feature_names=["Readability_Score_WienerSachtextformel-1_de",
    ...                                      "OtherFeature_de"])
    """

    # Ensure we have instances; if a class is passed we instantiate it.
    instances = []
    for ex in extractors:
        if isinstance(ex, type):
            instances.append(ex())
        else:
            instances.append(ex)

    def _run(cas: Any) -> None:
        for inst in instances:
            # Each extractor is expected to have an ``extract`` method.
            inst.extract(cas)

    return _run


def get_corpus_slug(folder: Path) -> str:
    """Return a normalized slug for a corpus folder.

    The slug is derived from the full folder path with path separators replaced
    by hyphens. This ensures a unique cache directory name for each corpus.
    """
    return str(folder).replace(os.sep, "-")

def _extract_scores_from_cas(cas, feature_name: str) -> list[float]:
    for feature in cas.select("org.lift.type.FeatureAnnotationNumeric"):        
        if feature.get('name') == feature_name:
            try:
                return [float(feature.value)]
            except Exception as e:
                raise RuntimeError(
                    f"Failed to convert feature value {feature.value!r}: {e}"
                )
    return []

def _process_file(
    file: Path,
    use_cache: bool,
    cache_dir: Path | None,
    *,
    # Callable that receives a CAS and performs one or more extractions.
    # By default it runs the original Wiener‑Sachtextformel extractor.
    feature_extractor: Callable[[Any], None],
    # List of feature annotation names that the extractor(s) will populate.
    feature_names: List[str],
) -> Dict[str, List[float]]:
    """Process a single *file* and return its readability scores.

    Parameters
    ----------
    file: Path
        The text file to process.
    use_cache: bool
        Whether to use a cached ``.xmi`` representation.
    cache_dir: Path | None
        Directory where cached ``.xmi`` files are stored (if caching is enabled).
    feature_extractor: Callable[[], object], optional
        A zero‑argument callable that returns an instance of a ``py_lift`` feature
        extractor. Defaults to :class:`FE_TextstatWienerSachtextformel_1`.
    feature_name: str, optional
        The name of the feature annotation to look for inside the CAS. This must
        match the ``name`` attribute used by the extractor. Defaults to the
        Wiener‑Sachtextformel‑1 readability feature.

    Returns
    -------
    list[float]
        A list containing the extracted score(s) for the given file.
    """
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

        feature_extractor(cas)

        result: Dict[str, List[float]] = {}
        for name in feature_names:
            result[name] = _extract_scores_from_cas(cas, name)
        return result
    except Exception as e:
        raise RuntimeError(f"Error processing file '{file}': {e}") from e


def process_folder(
    dir: Path,
    use_cache: bool = True,
    *,
    # Callable that receives a CAS and performs one or more extractions.
    feature_extractor: Callable[[Any], None] = lambda cas: FE_TextstatWienerSachtextformel_1().extract(cas),
    # List of feature annotation names to collect.
    feature_names: List[str] = ["Readability_Score_WienerSachtextformel-1_de"],
) -> Dict[str, List[float]]:
    """Process *dir* and return all readability scores (or other feature scores).

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

    # Initialise an aggregated dict with empty lists for each feature.
    aggregated: Dict[str, List[float]] = {name: [] for name in feature_names}
    for file in tqdm(dir.iterdir(), desc=f"Processing {dir.name}"):
        if not file.is_file():
            continue
        # Get per‑file dict of feature → scores.
        file_result = _process_file(
            file,
            use_cache,
            cache_dir if use_cache else None,
            feature_extractor=feature_extractor,
            feature_names=feature_names,
        )
        # Append each feature's scores to the aggregated dict.
        for name, vals in file_result.items():
            aggregated[name].extend(vals)
    return aggregated
