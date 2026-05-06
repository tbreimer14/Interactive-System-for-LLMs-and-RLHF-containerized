"""
ui/state.py

Streamlit session state management.

Centralises all st.session_state keys so components never write raw keys
themselves — they call helpers here instead.

Key groups:
    Traits / reward:   KEY_TRAITS, KEY_SAVED
    History:           KEY_HISTORY, KEY_SELECTED_HISTORY
    Dataset browser:   KEY_DATASET_CATEGORY, KEY_DATASET_INDEX, KEY_DATASET_POSTS
    UI:                KEY_NUM_RESPONSES
    Sessions:          KEY_ACTIVE_SESSION
    Online training:   KEY_CURRENT_ROUND, KEY_GRADING_BUFFER, KEY_ONLINE_SESSION

Online training state lifecycle:
    init_current_round(prompt, n)   called when "Generate Responses" is clicked
    add_to_current_round(text, r)   called after each "Save Response X"
    is_current_round_complete()     True when all n responses are saved
    pop_current_round()             returns and clears the completed round
    add_to_grading_buffer(round)    accumulates completed rounds (max 3)
    clear_grading_buffer()          called after OnlineGRPOSession.step() fires
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
KEY_ACTIVE_SESSION    = "active_session"     # str — current session/experiment name
KEY_RESPONSE_LENGTH    = "response_length"     # int  — target word count appended to generation prompt
KEY_CURRENT_ROUND      = "current_round"       # dict | None — in-progress grading round
KEY_GRADING_BUFFER     = "grading_buffer"      # list[dict] — completed rounds awaiting training
KEY_ONLINE_SESSION     = "online_session"      # OnlineGRPOSession | None
KEY_WEIGHTS_LOADED     = "weights_loaded"      # bool — True once checkpoint auto-load has run


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
    if KEY_ACTIVE_SESSION not in st.session_state:
        st.session_state[KEY_ACTIVE_SESSION] = "default"
    if KEY_RESPONSE_LENGTH not in st.session_state:
        st.session_state[KEY_RESPONSE_LENGTH] = 150
    if KEY_CURRENT_ROUND not in st.session_state:
        st.session_state[KEY_CURRENT_ROUND] = None
    if KEY_GRADING_BUFFER not in st.session_state:
        st.session_state[KEY_GRADING_BUFFER] = []
    if KEY_ONLINE_SESSION not in st.session_state:
        st.session_state[KEY_ONLINE_SESSION] = None
    if KEY_WEIGHTS_LOADED not in st.session_state:
        st.session_state[KEY_WEIGHTS_LOADED] = False


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


# ── Active session ─────────────────────────────────────────────────────────────

def get_active_session() -> str:
    return st.session_state.get(KEY_ACTIVE_SESSION, "default")


def get_response_length() -> int:
    return st.session_state.get(KEY_RESPONSE_LENGTH, 150)


def set_response_length(n: int) -> None:
    st.session_state[KEY_RESPONSE_LENGTH] = n


# ── Online round buffer ────────────────────────────────────────────────────────

def init_current_round(prompt: str, expected: int) -> None:
    st.session_state[KEY_CURRENT_ROUND] = {
        "prompt": prompt,
        "expected": expected,
        "completions": [],
    }


def add_to_current_round(text: str, reward: float) -> None:
    r = st.session_state.get(KEY_CURRENT_ROUND)
    if r is None:
        return
    r["completions"].append({"text": text, "reward": reward})


def is_current_round_complete() -> bool:
    r = st.session_state.get(KEY_CURRENT_ROUND)
    if r is None:
        return False
    return len(r["completions"]) >= r["expected"]


def pop_current_round() -> dict | None:
    r = st.session_state.get(KEY_CURRENT_ROUND)
    st.session_state[KEY_CURRENT_ROUND] = None
    return r


def get_grading_buffer() -> list:
    return st.session_state.get(KEY_GRADING_BUFFER, [])


def add_to_grading_buffer(round_data: dict) -> None:
    st.session_state[KEY_GRADING_BUFFER].append({
        "prompt": round_data["prompt"],
        "completions": round_data["completions"],
    })


def clear_grading_buffer() -> None:
    st.session_state[KEY_GRADING_BUFFER] = []


def get_online_session():
    return st.session_state.get(KEY_ONLINE_SESSION)


def set_online_session(session) -> None:
    st.session_state[KEY_ONLINE_SESSION] = session


def is_weights_loaded() -> bool:
    return st.session_state.get(KEY_WEIGHTS_LOADED, False)


def mark_weights_loaded() -> None:
    st.session_state[KEY_WEIGHTS_LOADED] = True


# ── Active session ─────────────────────────────────────────────────────────────

def switch_session(name: str) -> None:
    """Switch to a different session, resetting all session-scoped state."""
    st.session_state[KEY_ACTIVE_SESSION]    = name
    st.session_state[KEY_HISTORY]           = []
    st.session_state[KEY_SAVED]             = False
    st.session_state[KEY_CURRENT_ROUND]     = None
    st.session_state[KEY_GRADING_BUFFER]    = []
    st.session_state[KEY_ONLINE_SESSION]    = None
    st.session_state[KEY_WEIGHTS_LOADED]    = False
