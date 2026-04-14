"""
app.py

Entry point for the RLHF research UI.

Run with:
    uv run streamlit run app.py

This file is intentionally thin — all logic lives in:
    backend_adapter.py  (backend calls)
    ui/components.py    (panel rendering)
    ui/state.py         (session state)
    ui/storage.py       (JSONL read/write)
    ui/types.py         (data models)
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import streamlit as st

from backend_adapter import BackendAdapter
from app.trait_manager import load_traits
from ui.types import TraitConfig
from ui.storage import load_interactions
import ui.state as state
import ui.components as components

# ── Paths ──────────────────────────────────────────────────────────────────────
ROOT     = Path(__file__).parent
CONFIG   = ROOT / "config" / "traits.json"
LOG_PATH = str(ROOT / "logs" / "interaction_log.jsonl")

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(page_title="Interactive RLHF System", layout="wide")
st.title("Interactive RLHF System")

# ── Init ───────────────────────────────────────────────────────────────────────
default_traits = [TraitConfig.from_dict(t) for t in load_traits(str(CONFIG))]
state.init(default_traits)

# Load history from disk on first run
if not state.get_history():
    state.set_history(load_interactions(log_path=LOG_PATH))

adapter = BackendAdapter()

# ── Layout: main (left) + history (right) ─────────────────────────────────────
col_main, col_history = st.columns([2, 1])

with col_history:
    components.history_panel()

with col_main:
    components.prompt_panel(adapter)
    components.context_panel()

    if state.get_rag_result():
        components.traits_editor()
        scored_traits, scalar_reward = components.scoring_panel(adapter)
        components.save_panel(scored_traits, scalar_reward, log_path=LOG_PATH)
