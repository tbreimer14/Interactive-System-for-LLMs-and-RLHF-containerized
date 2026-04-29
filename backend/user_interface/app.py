"""
app.py

Entry point for the RLHF research UI.

Run with:
    uv run streamlit run app.py

This file is intentionally thin — all logic lives in:
    backend_adapter.py       (reward computation + model generation)
    ui/components.py         (panel rendering)
    ui/state.py              (session state)
    ui/session_manager.py    (experiment session isolation)
    ui/storage.py            (JSONL read/write)
    ui/types.py              (data models)
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import streamlit as st

from backend_adapter import BackendAdapter
from app.trait_manager import load_traits
from ui.types import TraitConfig
from ui.storage import load_interactions
from ui.session_manager import (
    list_sessions, ensure_session, is_valid_name,
    interaction_log_path, grpo_output_dir, DEFAULT_SESSION,
)
import ui.state as state
import ui.components as components

# ── Static paths ───────────────────────────────────────────────────────────────
ROOT   = Path(__file__).parent
CONFIG = ROOT / "config" / "traits.json"

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(page_title="Interactive RLHF System", layout="wide")

# ── Init ───────────────────────────────────────────────────────────────────────
default_traits = [TraitConfig.from_dict(t) for t in load_traits(str(CONFIG))]
state.init(default_traits)
adapter = BackendAdapter()

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("RLHF System")

    # Session switcher
    st.caption("Experiment Session")
    sessions       = list_sessions()
    active_session = state.get_active_session()
    if active_session not in sessions:
        sessions = sorted(sessions + [active_session])

    selected = st.selectbox(
        "session",
        options=sessions,
        index=sessions.index(active_session),
        label_visibility="collapsed",
    )
    if selected != active_session:
        ensure_session(selected)
        state.switch_session(selected)
        st.rerun()

    with st.expander("New session", expanded=False):
        new_name = st.text_input(
            "Name", key="new_session_name",
            placeholder="e.g. run_v2 or warm_tone",
        )
        if st.button("Create & Switch", use_container_width=True):
            name = new_name.strip()
            if not name:
                st.warning("Enter a session name.")
            elif not is_valid_name(name):
                st.warning("Only letters, numbers, _ and - allowed.")
            elif name in sessions:
                st.warning("Session already exists.")
            else:
                ensure_session(name)
                state.switch_session(name)
                st.rerun()

    st.divider()
    page = st.radio(
        "Navigate",
        ["Setup", "Prompt", "Grade", "History", "Analytics", "Train"],
        label_visibility="collapsed",
    )

# ── Session-specific paths ─────────────────────────────────────────────────────
active_session = state.get_active_session()
ensure_session(active_session)
LOG_PATH = interaction_log_path(active_session)
GRPO_DIR = grpo_output_dir(active_session)

# Load history for this session on first visit or after session switch
if not state.get_history():
    state.set_history(load_interactions(log_path=LOG_PATH))

# ── Route to page ──────────────────────────────────────────────────────────────
if page == "Setup":
    components.setup_page()
elif page == "Prompt":
    components.prompt_page()
elif page == "Grade":
    components.score_page(adapter, log_path=LOG_PATH)
elif page == "History":
    components.history_page()
elif page == "Analytics":
    components.analytics_page()
elif page == "Train":
    components.train_page(log_path=LOG_PATH, grpo_output_dir=GRPO_DIR)
