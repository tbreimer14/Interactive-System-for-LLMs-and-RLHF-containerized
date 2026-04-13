"""
ui/state.py

Streamlit session state management.

Centralises all st.session_state keys in one place so components
never write raw keys themselves — they call helpers here instead.
This makes it easy to see all shared state at a glance and rename
keys without hunting through components.py.
"""

import streamlit as st

from ui.types import TraitConfig


# ── Keys ──────────────────────────────────────────────────────────────────────
# Define all session state keys as constants to avoid typos across files.

KEY_TRAITS      = "traits"        # list[TraitConfig]
KEY_RAG_RESULT  = "rag_result"    # dict | None  (from BackendAdapter.generate)
KEY_SAVED       = "saved"         # bool — True once current interaction is saved
KEY_HISTORY     = "history"       # list[dict]   (loaded from JSONL at startup)


def init(default_traits: list[TraitConfig]) -> None:
    """
    Initialise all session state keys on first run.

    Streamlit reruns this file on every interaction, but the `if key not in`
    guard ensures we only set defaults once per browser session.

    Args:
        default_traits: traits loaded from config/traits.json at startup
    """
    if KEY_TRAITS not in st.session_state:
        st.session_state[KEY_TRAITS] = default_traits

    if KEY_RAG_RESULT not in st.session_state:
        st.session_state[KEY_RAG_RESULT] = None

    if KEY_SAVED not in st.session_state:
        st.session_state[KEY_SAVED] = False

    if KEY_HISTORY not in st.session_state:
        st.session_state[KEY_HISTORY] = []


# ── Accessors ─────────────────────────────────────────────────────────────────

def get_traits() -> list[TraitConfig]:
    return st.session_state[KEY_TRAITS]


def set_traits(traits: list[TraitConfig]) -> None:
    st.session_state[KEY_TRAITS] = traits


def get_rag_result() -> dict | None:
    return st.session_state[KEY_RAG_RESULT]


def set_rag_result(result: dict) -> None:
    st.session_state[KEY_RAG_RESULT] = result
    st.session_state[KEY_SAVED] = False  # new result always starts unsaved


def is_saved() -> bool:
    return st.session_state[KEY_SAVED]


def mark_saved() -> None:
    st.session_state[KEY_SAVED] = True


def clear() -> None:
    """Reset to a blank slate (used by the Clear button)."""
    st.session_state[KEY_RAG_RESULT] = None
    st.session_state[KEY_SAVED] = False


def get_history() -> list[dict]:
    return st.session_state[KEY_HISTORY]


def set_history(entries: list[dict]) -> None:
    st.session_state[KEY_HISTORY] = entries


def append_history(entry: dict) -> None:
    st.session_state[KEY_HISTORY].append(entry)
