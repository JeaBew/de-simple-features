"""Master runner that executes all analysis scripts.

Each individual ``run_*.py`` module defines a ``main`` function that performs a
specific analysis (e.g., token length, frequency ratios, readability, etc.).
This script imports those modules and invokes their ``main`` functions in a
deterministic order so that a single command runs the full suite of analyses.
"""

# Import the individual runner modules. The imports are placed at the top so
# that any import‑time errors are raised early, making debugging easier.
import run_token_length
import run_negations
import run_freq
import run_readability
import run_sent_length


def main() -> None:
    run_token_length.main()
    run_sent_length.main()
    run_readability.main()
    run_freq.main()
#    run_negations.main()

if __name__ == "__main__":
    main()