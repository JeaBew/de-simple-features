import pandas as pd
import os
from pathlib import Path
from statistics import mean

from paper_easy_plain_german_datasets.utils import process_folder

CRAWLED_ORIG = Path("data/crawled/orig")
CRAWLED_PLAIN = Path("data/crawled/plain")


def main() -> None:
    # Gather all readability scores from both corpora using the shared utility.
    scores_crawled_orig = process_folder(CRAWLED_ORIG)
    scores_crawled_plain = process_folder(CRAWLED_PLAIN)

    # Compute and output the mean readability score for each corpus.
    mean_orig = mean(scores_crawled_orig) if scores_crawled_orig else float('nan')
    mean_plain = mean(scores_crawled_plain) if scores_crawled_plain else float('nan')
    print(f"Mean readability (orig): {mean_orig}")
    print(f"Mean readability (plain): {mean_plain}")

if __name__ == "__main__":
    main()
