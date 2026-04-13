"""
ui/components.py

One rendering function per UI panel.

Each function is self-contained:
  - reads from session state via ui.state helpers
  - writes back to session state via ui.state helpers
  - calls the backend only through the BackendAdapter passed in
  - never imports st directly from outside this file (keeps mock-testing easy)

Panel functions:
    prompt_panel(adapter)       -> renders panel 1 + 2
    context_panel()             -> renders panel 3 (collapsible)
    traits_editor()             -> renders panel 4
    scoring_panel(adapter)      -> renders panel 5, returns (scored_traits, scalar_reward)
    save_panel(...)             -> renders panel 6
    history_panel()             -> renders panel 7
"""

from datetime import datetime

import streamlit as st

import ui.state as state
from ui.types import TraitConfig, InteractionLog
from ui.storage import save_interaction


# ==============================================================================
# Panel 1 + 2: Prompt input and response display
# ==============================================================================

def prompt_panel(adapter) -> None:
    """
    Render the prompt text area, Generate and Clear buttons,
    and the response display.

    Calls adapter.generate() on submit and writes result to session state.
    """
    st.subheader("1. Prompt")

    prompt = st.text_area(
        "prompt",
        height=120,
        placeholder="Type a question or post here...",
        label_visibility="collapsed",
    )

    col_gen, col_clr = st.columns([1, 1])

    with col_gen:
        if st.button("Generate Response", type="primary", use_container_width=True):
            if not prompt.strip():
                st.warning("Please enter a prompt first.")
            else:
                with st.spinner("Generating..."):
                    try:
                        result = adapter.generate(prompt.strip(), top_k=3)
                        state.set_rag_result(result)
                    except Exception as e:
                        st.error(f"Generation failed: {e}")

    with col_clr:
        if st.button("Clear", use_container_width=True):
            state.clear()
            st.rerun()

    # Response display
    rag = state.get_rag_result()
    if rag:
        st.subheader("2. Response")
        st.markdown(rag["answer"])


# ==============================================================================
# Panel 3: Retrieved context (collapsible, read-only)
# ==============================================================================

def context_panel() -> None:
    """
    Render the collapsible retrieved context panel.
    Only shown when a RAG result is available.
    """
    rag = state.get_rag_result()
    if not rag:
        return

    with st.expander("3. Retrieved context", expanded=False):
        chunks = rag.get("retrieved", [])
        if not chunks:
            st.caption("No chunks retrieved.")
        for i, chunk in enumerate(chunks):
            st.markdown(f"**Chunk {i + 1}** — `{chunk.get('source', 'unknown')}`")
            st.text(chunk.get("text", ""))
            if i < len(chunks) - 1:
                st.divider()


# ==============================================================================
# Panel 4: Traits editor
# ==============================================================================

def traits_editor() -> None:
    """
    Render the dynamic traits editor.

    Allows the user to:
      - add a new trait (name, description, weight)
      - remove an existing trait
      - edit the weight of any trait in-place
    """
    st.subheader("4. Traits")

    traits = state.get_traits()

    # Add trait form
    with st.expander("Add a trait", expanded=False):
        new_name   = st.text_input("Trait name",  key="add_name")
        new_desc   = st.text_input("Description", key="add_desc")
        new_weight = st.number_input("Weight", min_value=0.0, step=0.1,
                                     value=0.5, key="add_weight")
        if st.button("Add trait"):
            if not new_name.strip():
                st.warning("Trait name cannot be empty.")
            else:
                traits.append(TraitConfig(
                    name=new_name.strip(),
                    description=new_desc.strip(),
                    weight=new_weight,
                ))
                state.set_traits(traits)
                st.rerun()

    if not traits:
        st.caption("No traits defined. Add one above.")
        return

    # Editable trait rows
    updated = []
    for i, trait in enumerate(traits):
        col_label, col_weight, col_remove = st.columns([3, 1, 1])
        with col_label:
            st.markdown(f"**{trait.name}**" + (f" — _{trait.description}_" if trait.description else ""))
        with col_weight:
            new_w = st.number_input(
                "weight",
                min_value=0.0, step=0.1,
                value=float(trait.weight),
                key=f"trait_weight_{i}",
                label_visibility="collapsed",
            )
            trait = TraitConfig(name=trait.name, description=trait.description, weight=new_w)
        with col_remove:
            if st.button("Remove", key=f"remove_trait_{i}"):
                state.set_traits([t for j, t in enumerate(traits) if j != i])
                st.rerun()
        updated.append(trait)

    state.set_traits(updated)


# ==============================================================================
# Panel 5: Response scoring
# ==============================================================================

def scoring_panel(adapter) -> tuple:
    """
    Render a slider per trait, compute scalar reward live, display breakdown.

    Returns:
        (scored_traits, scalar_reward) or (None, None) if no traits / no result
    """
    if not state.get_rag_result():
        return None, None

    traits = state.get_traits()
    if not traits:
        return None, None

    st.subheader("5. Score this response")

    scores = {}
    for trait in traits:
        scores[trait.name] = st.slider(
            f"{trait.name}  (weight {trait.weight})",
            min_value=0, max_value=5, value=3,
            key=f"score_{trait.name}",
        )

    try:
        scored_traits, scalar_reward = adapter.compute_reward(scores, traits)
    except Exception as e:
        st.error(f"Reward computation failed: {e}")
        return None, None

    # Breakdown display
    st.markdown("**Reward breakdown:**")
    cols = st.columns(len(traits) + 1)
    for j, st_trait in enumerate(scored_traits):
        with cols[j]:
            st.metric(
                label=st_trait.name,
                value=f"{st_trait.score} / 5",
                delta=f"x{st_trait.weight} = {st_trait.contribution:.3f}",
            )
    with cols[-1]:
        st.metric(label="Scalar reward", value=f"{scalar_reward:.3f}")

    return scored_traits, scalar_reward


# ==============================================================================
# Panel 6: Save interaction
# ==============================================================================

def save_panel(scored_traits, scalar_reward, log_path: str) -> None:
    """
    Render the Save button.

    Builds an InteractionLog from current session state + scoring results,
    writes it to JSONL, and marks the interaction as saved.
    """
    if not state.get_rag_result() or scored_traits is None:
        return

    st.subheader("6. Save")

    if state.is_saved():
        st.success("Interaction saved to log.")
        return

    if st.button("Save interaction", type="primary"):
        rag = state.get_rag_result()
        entry = InteractionLog(
            timestamp=datetime.now().isoformat(timespec="seconds"),
            prompt=rag["query"],
            retrieved_chunks=rag["retrieved"],
            response=rag["answer"],
            traits=[st.to_dict() for st in scored_traits],
            scalar_reward=scalar_reward,
        )
        try:
            save_interaction(entry, log_path=log_path)
            state.mark_saved()
            state.append_history(entry.to_dict())
            st.rerun()
        except Exception as e:
            st.error(f"Failed to save: {e}")


# ==============================================================================
# Panel 7: History
# ==============================================================================

def history_panel() -> None:
    """
    Render the history side panel.

    Shows past saved interactions loaded from JSONL at startup,
    plus any saved in the current session.
    """
    st.subheader("History")

    entries = list(reversed(state.get_history()))

    if not entries:
        st.caption("No saved interactions yet.")
        return

    for entry in entries:
        reward = entry.get("scalar_reward", "?")
        reward_str = f"{reward:.2f}" if isinstance(reward, (int, float)) else str(reward)
        ts    = entry.get("timestamp", "")[:16]
        short_prompt = entry.get("prompt", "")[:40]

        with st.expander(f"[{reward_str}]  {ts}  — {short_prompt}...", expanded=False):
            st.markdown(f"**Prompt:** {entry.get('prompt', '')}")
            st.markdown(f"**Response:** {entry.get('response', '')[:300]}...")
            st.markdown(f"**Scalar reward:** `{reward_str}`")

            traits_data = entry.get("traits", [])
            if traits_data:
                st.markdown("**Traits:**")
                for t in traits_data:
                    st.markdown(
                        f"- {t.get('name')}  score={t.get('score')}  "
                        f"weight={t.get('weight')}  "
                        f"contribution={t.get('contribution')}"
                    )
