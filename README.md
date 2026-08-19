# Linguistic Features of German Easy and Plain Language

Code for the paper **"Linguistic Features of German Easy and Plain Language: A Cross-Dataset Analysis"**.

This repository contains the scripts used to compute and visualize linguistic features for German Standard Language, Plain Language, and Easy Language corpora. The analysis is corpus-level and descriptive: the datasets are heterogeneous and not uniformly parallel, so the results should not be interpreted as causal effects of simplification.

## Features

The scripts compute surface-level and guideline-related indicators, including:

- average token length
- average sentence length
- vocabulary-frequency bands
- readability scores using the Wiener Sachtextformel
- document-level presence of subjunctive forms
- document-level presence of selected negation forms

More features will be added in the future.

## Data

The datasets are not included and have to be downloaded manually. Scripts for data preparation are provided in the `preparation/` directory. Please make sure that the paths in `constants.py` match your local data locations.

The analysis uses the following public resources:

- **TIGER corpus** — Brants et al. (2004)
- **ASGC: Aligned Simple German Corpus** — Toborek et al. (2023)
- **DEplain-web** — Stodden et al. (2023)
- **German4All** — Anschütz et al. (2025)
- **Leiko** — Jablotschkin and Zinsmeister (2024)

Please cite the original dataset papers when using these resources.

## References

Please see the paper for full bibliographic references to the datasets and tools used in this repository.

## Usage

To run all implemented analyses, execute:

python3 run_all.py
