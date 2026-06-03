"""Load the KaggleDS task-based dataset-search corpus from the HuggingFace Hub.

The corpus (train / validation / test splits) is published at:
    https://huggingface.co/datasets/trl-lab/kaggleds-corpus-task-based-search-bench

The processed CSV splits (~300 MB) are intentionally NOT vendored into this
repository. Fetch them on demand with this loader instead.

Example:
    from evaluation.data.load_kaggleds import load_kaggleds

    test = load_kaggleds("test")          # a single split
    corpus = load_kaggleds()              # all splits (DatasetDict)
"""

from datasets import load_dataset

HF_REPO_ID = "trl-lab/kaggleds-corpus-task-based-search-bench"


def load_kaggleds(split=None, **kwargs):
    """Load the KaggleDS corpus from the HuggingFace Hub.

    Args:
        split: One of "train", "validation", "test", or None for all splits.
        **kwargs: Forwarded to ``datasets.load_dataset`` (e.g. ``cache_dir``).

    Returns:
        A ``datasets.Dataset`` if ``split`` is given, otherwise a
        ``datasets.DatasetDict`` containing all splits.
    """
    return load_dataset(HF_REPO_ID, split=split, **kwargs)


if __name__ == "__main__":
    print(load_kaggleds())
