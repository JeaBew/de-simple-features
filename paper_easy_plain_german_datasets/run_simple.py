import pandas as pd
import os
from pathlib import Path
from statistics import mean

from utils import process_folder

CRAWLED_ORIG = Path("data/crawled/orig")
CRAWLED_PLAIN = Path("data/crawled/plain")
CRAWLED_EASY = Path("data/crawled/easy")

GFA_ORIG = Path("data/german4all/orig")
GFA_PLAIN = Path("data/german4all/plain")

def main() -> None:
    # Gather all readability scores from both corpora using the shared utility.
    scores_crawled_orig = process_folder(CRAWLED_ORIG)
    scores_crawled_plain = process_folder(CRAWLED_PLAIN)
    scores_crawled_easy = process_folder(CRAWLED_EASY)

    # Compute and output the mean readability score for each corpus.
    mean_orig = mean(scores_crawled_orig) if scores_crawled_orig else float('nan')
    mean_plain = mean(scores_crawled_plain) if scores_crawled_plain else float('nan')
    mean_easy = mean(scores_crawled_easy) if scores_crawled_easy else float('nan')
    print(f"Mean readability (orig): {mean_orig}")
    print(f"Mean readability (plain): {mean_plain}")
    print(f"Mean readability (easy): {mean_easy}")

    # Gather all readability scores from both corpora using the shared utility.
    scores_gfa_orig = process_folder(GFA_ORIG)
    scores_gfa_plain = process_folder(GFA_PLAIN)

    # Compute and output the mean readability score for each corpus.
    mean_orig = mean(scores_gfa_orig) if scores_gfa_orig else float('nan')
    mean_plain = mean(scores_gfa_plain) if scores_gfa_plain else float('nan')
    print(f"Mean readability (orig): {mean_orig}")
    print(f"Mean readability (plain): {mean_plain}")

if __name__ == "__main__":
    main()
