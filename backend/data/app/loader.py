"""
Data loader module for the 20 Newsgroups dataset.

Functions:
- load_20newsgroups(categories, subset, remove): Fetch dataset from sklearn
- get_category_names(): Return all 20 newsgroup category names
"""

from sklearn.datasets import fetch_20newsgroups


ALL_CATEGORIES = [
    "alt.atheism",
    "comp.graphics",
    "comp.os.ms-windows.misc",
    "comp.sys.ibm.pc.hardware",
    "comp.sys.mac.hardware",
    "comp.windows.x",
    "misc.forsale",
    "rec.autos",
    "rec.motorcycles",
    "rec.sport.baseball",
    "rec.sport.hockey",
    "sci.crypt",
    "sci.electronics",
    "sci.med",
    "sci.space",
    "soc.religion.christian",
    "talk.politics.guns",
    "talk.politics.mideast",
    "talk.politics.misc",
    "talk.religion.misc",
]


def load_20newsgroups(categories=None, subset="all", remove=("headers", "footers", "quotes")):
    """
    Fetch the 20 Newsgroups dataset from sklearn.

    Args:
        categories (list[str] | None): Newsgroup categories to load. None loads all 20.
        subset (str): "train", "test", or "all".
        remove (tuple[str]): Parts of posts to strip — any of "headers", "footers", "quotes".

    Returns:
        dict: {
            "data": list[str],         # raw post texts
            "target": list[int],       # category index per post
            "target_names": list[str], # category name per index
            "filenames": list[str],    # original file paths
        }
    """
    dataset = fetch_20newsgroups(
        subset=subset,
        categories=categories,
        remove=remove,
        shuffle=False,
    )
    return {
        "data": dataset.data,
        "target": list(dataset.target),
        "target_names": list(dataset.target_names),
        "filenames": list(dataset.filenames),
    }


def get_category_names():
    """Return the full list of 20 newsgroup category names."""
    return ALL_CATEGORIES
