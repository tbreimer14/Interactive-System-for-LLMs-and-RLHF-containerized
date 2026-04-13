"""
main.py

Streamlit UI for the RLHF research prototype.

Run with:
    uv run streamlit run main.py

Panels:
    1. Prompt input + Generate / Clear
    2. Response display
    3. Retrieved context (collapsible)
    4. Traits editor (add / remove / reweight)
    5. Scoring panel -> scalar reward
    6. Save interaction
    7. History
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import streamlit as st

from app.rag_stub import answer
from app.trait_manager import load_traits
from app.reward_bridge import compute_reward
from app.interaction_logger import log_interaction, read_log

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT       = Path(__file__).parent
CONFIG     = ROOT / "config" / "traits.json"
LOG_PATH   = ROOT / "logs" / "interaction_log.jsonl"

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(page_title="RLHF Interface", layout="wide")
st.title("RLHF Research Interface")

# ── Session state init ─────────────────────────────────────────────────────────
if "traits" not in st.session_state:
    st.session_state.traits = load_traits(str(CONFIG))

if "rag_result" not in st.session_state:
    st.session_state.rag_result = None  # set after generation

if "saved" not in st.session_state:
    st.session_state.saved = False  # tracks if current interaction was saved

# ── Layout: left column (main) + right column (history) ───────────────────────
col_main, col_history = st.columns([2, 1])

# ==============================================================================
# RIGHT: History panel
# ==============================================================================
with col_history:
    st.subheader("History")

    entries = read_log(log_path=str(LOG_PATH))

    if not entries:
        st.caption("No saved interactions yet.")
    else:
        for i, entry in enumerate(reversed(entries)):
            reward_str = f"{entry.get('scalar_reward', '?'):.2f}" \
                if isinstance(entry.get('scalar_reward'), (int, float)) else "?"
            with st.expander(
                f"[{reward_str}]  {entry.get('timestamp', '')[:16]}  "
                f"— {entry.get('prompt', '')[:40]}...",
                expanded=False,
            ):
                st.markdown(f"**Prompt:** {entry.get('prompt', '')}")
                st.markdown(f"**Response:** {entry.get('response', '')[:200]}...")
                st.markdown(f"**Scalar reward:** `{entry.get('scalar_reward', '?')}`")
                st.markdown(f"**Scores:** {entry.get('scores', {})}")

# ==============================================================================
# LEFT: Main workflow
# ==============================================================================
with col_main:

    # ── 1. Prompt input ────────────────────────────────────────────────────────
    st.subheader("1. Prompt")

    prompt = st.text_area(
        "Enter your prompt",
        height=120,
        placeholder="Type a question or post here...",
        label_visibility="collapsed",
    )

    btn_generate, btn_clear = st.columns([1, 1])

    with btn_generate:
        generate_clicked = st.button("Generate Response", type="primary", use_container_width=True)

    with btn_clear:
        if st.button("Clear", use_container_width=True):
            st.session_state.rag_result = None
            st.session_state.saved = False
            st.rerun()

    # Run RAG on generate
    if generate_clicked and prompt.strip():
        with st.spinner("Generating..."):
            st.session_state.rag_result = answer(prompt.strip(), k=3)
            st.session_state.saved = False

    elif generate_clicked and not prompt.strip():
        st.warning("Please enter a prompt first.")

    # ── 2. Response display ────────────────────────────────────────────────────
    if st.session_state.rag_result:
        result = st.session_state.rag_result

        st.subheader("2. Response")
        st.markdown(result["answer"])

        # ── 3. Retrieved context (collapsible) ─────────────────────────────────
        with st.expander("3. Retrieved context", expanded=False):
            for i, chunk in enumerate(result["retrieved"]):
                st.markdown(f"**Chunk {i + 1}** — `{chunk['source']}`")
                st.text(chunk["text"])
                st.divider()

        # ── 4. Traits editor ───────────────────────────────────────────────────
        st.subheader("4. Traits")

        traits = st.session_state.traits

        # Add trait
        with st.expander("Add a trait", expanded=False):
            new_name  = st.text_input("Trait name",        key="new_name")
            new_desc  = st.text_input("Description",       key="new_desc")
            new_weight = st.number_input("Weight", min_value=0.0, step=0.1,
                                         value=0.5,        key="new_weight")
            if st.button("Add trait"):
                if new_name.strip():
                    traits.append({
                        "name":        new_name.strip(),
                        "description": new_desc.strip(),
                        "weight":      new_weight,
                    })
                    st.session_state.traits = traits
                    st.rerun()
                else:
                    st.warning("Trait name cannot be empty.")

        # Edit / remove existing traits
        updated_traits = []
        for i, trait in enumerate(traits):
            tcol_name, tcol_weight, tcol_remove = st.columns([3, 1, 1])
            with tcol_name:
                st.markdown(f"**{trait['name']}** — _{trait['description']}_")
            with tcol_weight:
                new_w = st.number_input(
                    "weight", min_value=0.0, step=0.1,
                    value=float(trait["weight"]),
                    key=f"weight_{i}",
                    label_visibility="collapsed",
                )
                trait = {**trait, "weight": new_w}
            with tcol_remove:
                if st.button("Remove", key=f"remove_{i}"):
                    st.session_state.traits = [
                        t for j, t in enumerate(traits) if j != i
                    ]
                    st.rerun()
            updated_traits.append(trait)

        st.session_state.traits = updated_traits

        # ── 5. Scoring panel ───────────────────────────────────────────────────
        st.subheader("5. Score this response")

        scores = {}
        for trait in st.session_state.traits:
            scores[trait["name"]] = st.slider(
                f"{trait['name']}  (weight {trait['weight']})",
                min_value=0, max_value=5, value=3,
                key=f"score_{trait['name']}",
            )

        # Compute and display reward breakdown
        if st.session_state.traits:
            reward_result = compute_reward(scores, st.session_state.traits)

            st.markdown("**Reward breakdown:**")
            cols = st.columns(len(st.session_state.traits) + 1)
            for j, trait in enumerate(st.session_state.traits):
                name = trait["name"]
                with cols[j]:
                    st.metric(
                        label=name,
                        value=f"{scores[name]} / 5",
                        delta=f"x{trait['weight']} = {reward_result['contributions'][name]:.3f}",
                    )
            with cols[-1]:
                st.metric(label="Scalar reward", value=f"{reward_result['scalar_reward']:.3f}")

        # ── 6. Save interaction ────────────────────────────────────────────────
        st.subheader("6. Save")

        if st.session_state.saved:
            st.success("Interaction saved to log.")
        else:
            if st.button("Save interaction", type="primary"):
                entry = {
                    "prompt":        result["query"],
                    "retrieved":     result["retrieved"],
                    "response":      result["answer"],
                    "traits":        st.session_state.traits,
                    "scores":        scores,
                    "scalar_reward": reward_result["scalar_reward"],
                }
                log_interaction(entry, log_path=str(LOG_PATH))
                st.session_state.saved = True
                st.rerun()
