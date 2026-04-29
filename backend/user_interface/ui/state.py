"""
ui/state.py

Streamlit session state management.

Centralises all st.session_state keys in one place so components
never write raw keys themselves — they call helpers here instead.
"""

import streamlit as st

from ui.types import TraitConfig


# ── Keys ──────────────────────────────────────────────────────────────────────

KEY_TRAITS           = "traits"           # list[TraitConfig]
KEY_SAVED            = "saved"            # bool — True once current interaction is saved
KEY_HISTORY          = "history"          # list[dict] (loaded from JSONL at startup)
KEY_SELECTED_HISTORY = "selected_history" # int | None (master-detail selection index)
KEY_DATASET_CATEGORY = "dataset_category" # str — selected newsgroup category
KEY_DATASET_INDEX    = "dataset_index"    # int — current post index within loaded dataset
KEY_DATASET_POSTS    = "dataset_posts"    # list[str] — cached posts for selected category
KEY_NUM_RESPONSES    = "num_responses"    # int — how many response slots to show (2 or 4)
KEY_GRPO_SESSION     = "grpo_session"    # GRPOSession | None — active or last training run
KEY_ACTIVE_SESSION   = "active_session"  # str — current session/experiment name


def init(default_traits: list[TraitConfig]) -> None:
    """Initialise all session state keys on first run."""
    if KEY_TRAITS not in st.session_state:
        st.session_state[KEY_TRAITS] = default_traits
    if KEY_SAVED not in st.session_state:
        st.session_state[KEY_SAVED] = False
    if KEY_HISTORY not in st.session_state:
        st.session_state[KEY_HISTORY] = []
    if KEY_SELECTED_HISTORY not in st.session_state:
        st.session_state[KEY_SELECTED_HISTORY] = None
    if KEY_DATASET_CATEGORY not in st.session_state:
        st.session_state[KEY_DATASET_CATEGORY] = None
    if KEY_DATASET_INDEX not in st.session_state:
        st.session_state[KEY_DATASET_INDEX] = 0
    if KEY_DATASET_POSTS not in st.session_state:
        st.session_state[KEY_DATASET_POSTS] = []
    if KEY_NUM_RESPONSES not in st.session_state:
        st.session_state[KEY_NUM_RESPONSES] = 2
    if KEY_GRPO_SESSION not in st.session_state:
        st.session_state[KEY_GRPO_SESSION] = None
    if KEY_ACTIVE_SESSION not in st.session_state:
        st.session_state[KEY_ACTIVE_SESSION] = "default"


# ── Accessors ─────────────────────────────────────────────────────────────────

def get_traits() -> list[TraitConfig]:
    return st.session_state[KEY_TRAITS]


def set_traits(traits: list[TraitConfig]) -> None:
    st.session_state[KEY_TRAITS] = traits


def is_saved() -> bool:
    return st.session_state[KEY_SAVED]


def mark_saved() -> None:
    st.session_state[KEY_SAVED] = True


def clear() -> None:
    """Reset scoring state and clear article + responses. Instruction is intentionally kept."""
    st.session_state[KEY_SAVED] = False
    if "article_input" in st.session_state:
        st.session_state["article_input"] = ""
    for i in range(4):
        for key in (f"response_input_{i}", f"saved_{i}"):
            if key in st.session_state:
                st.session_state[key] = "" if key.startswith("response") else False


def get_history() -> list[dict]:
    return st.session_state[KEY_HISTORY]


def set_history(entries: list[dict]) -> None:
    st.session_state[KEY_HISTORY] = entries


def append_history(entry: dict) -> None:
    st.session_state[KEY_HISTORY].append(entry)


def get_selected_history() -> int | None:
    return st.session_state.get(KEY_SELECTED_HISTORY)


def set_selected_history(idx: int | None) -> None:
    st.session_state[KEY_SELECTED_HISTORY] = idx


# ── Dataset browser state ─────────────────────────────────────────────────────

def get_dataset_category() -> str | None:
    return st.session_state.get(KEY_DATASET_CATEGORY)


def set_dataset_category(category: str | None) -> None:
    st.session_state[KEY_DATASET_CATEGORY] = category


def get_dataset_index() -> int:
    return st.session_state.get(KEY_DATASET_INDEX, 0)


def set_dataset_index(idx: int) -> None:
    st.session_state[KEY_DATASET_INDEX] = idx


def get_dataset_posts() -> list:
    return st.session_state.get(KEY_DATASET_POSTS, [])


def set_dataset_posts(posts: list) -> None:
    st.session_state[KEY_DATASET_POSTS] = posts


# ── Response count ─────────────────────────────────────────────────────────────

def get_num_responses() -> int:
    return st.session_state.get(KEY_NUM_RESPONSES, 2)


def set_num_responses(n: int) -> None:
    st.session_state[KEY_NUM_RESPONSES] = n


# ── GRPO training session ──────────────────────────────────────────────────────

def get_grpo_session():
    return st.session_state.get(KEY_GRPO_SESSION)


def set_grpo_session(session) -> None:
    st.session_state[KEY_GRPO_SESSION] = session


# ── Active session ─────────────────────────────────────────────────────────────

def get_active_session() -> str:
    return st.session_state.get(KEY_ACTIVE_SESSION, "default")


def switch_session(name: str) -> None:
    """Switch to a different session, resetting all session-scoped state."""
    st.session_state[KEY_ACTIVE_SESSION] = name
    st.session_state[KEY_GRPO_SESSION]   = None
    st.session_state[KEY_HISTORY]        = []
    st.session_state[KEY_SAVED]          = False
