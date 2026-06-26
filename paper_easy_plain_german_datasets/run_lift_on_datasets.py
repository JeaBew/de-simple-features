import pandas as pd
from cassis import cas_to_comparable_text
from py_lift.preprocessing import Spacy_Preprocessor
from py_lift.utils.core import load_lift_typesystem
# Reuse the readability extraction helper to avoid duplicated FE_TextstatFleschIndex calls.
from paper_easy_plain_german_datasets.utils import 
from pathlib import Path

prep = Spacy_Preprocessor("de", auto_install_models=True)
ts = load_lift_typesystem()

# ---------------------------------------------------------------------------
# Configuration for datasets
# Each entry defines how to read a CSV and where to store the lifted XMI files.
# ---------------------------------------------------------------------------
DATASET_CONFIG = [
    {
        "name": "deplain",
        "csv_path": "datasets_raw/DEPlain/deplain.csv",
        "text_column": "original",
        "simple_column": "simplification",
        "id_column": "pair_id",
        "output_subdir": "deplain",
        "prefix": "deplain",
    },
    {
        "name": "g4a",
        "csv_path": "datasets_raw/german4all_corrected/german4all_corrected.csv",
        "text_column": "text",
        "simple_column": "cl_LS",
        "id_column": "id",
        "output_subdir": "g4a",
        "prefix": "g4a",
    },
    {
        "name": "asgc",
        "csv_path": "datasets_raw/ASGC/to_csv/asgc.csv",
        "text_column": "original",
        "simple_column": "simple",
        "id_column": "id",
        "output_subdir": "asgc",
        "prefix": "asgc",
    },
]


def process_dataset(cfg: dict) -> None:
    """Lift a single dataset according to the provided configuration.

    The function reads the CSV, extracts the required columns, runs the
    preprocessing pipeline, adds the Flesch readability feature and writes the
    resulting CAS objects to XMI files.
    """
    df = pd.read_csv(cfg["csv_path"])
    # Show column information for debugging purposes
    print(df.columns.tolist())
    subset = df[[cfg["text_column"], cfg["simple_column"], cfg["id_column"]]]
    print(subset.head())

    output_dir_text = Path(f"datasets_lifted/{cfg['output_subdir']}/text")
    output_dir_simple = Path(f"datasets_lifted/{cfg['output_subdir']}/simple")
    # Ensure directories exist
    output_dir_text.mkdir(parents=True, exist_ok=True)
    output_dir_simple.mkdir(parents=True, exist_ok=True)

    for _, row in subset.iterrows():
        text = row[cfg["text_column"]]
        simple = row[cfg["simple_column"]]
        docid = str(row[cfg["id_column"]])

        cas_text = prep.run(text)
        cas_simple = prep.run(simple)

        # Apply readability extraction using the shared helper to avoid duplication.
        apply_readability(cas_text)
        apply_readability(cas_simple)

        outfile_text = output_dir_text / f"{cfg['prefix']}_text_{docid}.xmi"
        outfile_text.write_text(cas_text.to_xmi(), encoding="utf-8")

        outfile_simple = output_dir_simple / f"{cfg['prefix']}_simple_{docid}.xmi"
        outfile_simple.write_text(cas_simple.to_xmi(), encoding="utf-8")


def main() -> None:
    for cfg in DATASET_CONFIG:
        process_dataset(cfg)


if __name__ == "__main__":
    main()
