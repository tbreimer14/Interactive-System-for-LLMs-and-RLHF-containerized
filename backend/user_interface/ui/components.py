"""
ui/components.py

Page-level layout functions and reusable panel helpers.

Page functions (called from app.py):
    setup_page()                  -> Page 1: trait definition
    prompt_page()                 -> Page 2: dataset browser + prompt editing
    score_page(adapter, log_path) -> Page 3: response input + reward scoring
    history_page()                -> Page 4: master-detail history view
    analytics_page()              -> Page 5: tabbed reward analytics
    train_page(log_path)          -> Page 6: GRPO fine-tuning on saved interactions

Internal panel helpers (composed by page functions):
    _traits_editor()
    _dataset_loader()
    _dataset_picker()
    _scoring_panel(adapter)
    _save_panel(scored_traits, scalar_reward, prompt, response, log_path)
    _inject_column_resizer()
"""

import random
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import pandas as pd
import streamlit as st
import streamlit.components.v1 as _components
from sklearn.datasets import fetch_20newsgroups
from data.app.cleaner import clean_text

import ui.state as state
from ui.types import TraitConfig, InteractionLog
from ui.storage import save_interaction

_PRESET_INSTRUCTION_PATH = Path(__file__).parent.parent / "config" / "preset_instruction.txt"


def _load_preset_instruction() -> str:
    try:
        return _PRESET_INSTRUCTION_PATH.read_text(encoding="utf-8").strip()
    except FileNotFoundError:
        return ""


def _save_preset_instruction(text: str) -> None:
    _PRESET_INSTRUCTION_PATH.write_text(text.strip(), encoding="utf-8")

def _truncate_to_boundary(text: str, max_chars: int) -> str:
    """Truncate text to max_chars, breaking at the last paragraph or sentence boundary."""
    if len(text) <= max_chars:
        return text
    chunk = text[:max_chars]
    # Prefer paragraph boundary
    para = chunk.rfind("\n\n")
    if para > max_chars // 2:
        return chunk[:para].rstrip()
    # Fall back to sentence boundary (. ! ? followed by space or newline)
    best = -1
    for punct in (".", "!", "?"):
        for suffix in (" ", "\n"):
            pos = chunk.rfind(punct + suffix)
            if pos > best:
                best = pos
    if best > max_chars // 2:
        return chunk[:best + 1].rstrip()
    return chunk.rstrip()


_ALL_CATEGORIES = [
    "alt.atheism", "comp.graphics", "comp.os.ms-windows.misc",
    "comp.sys.ibm.pc.hardware", "comp.sys.mac.hardware", "comp.windows.x",
    "misc.forsale", "rec.autos", "rec.motorcycles", "rec.sport.baseball",
    "rec.sport.hockey", "sci.crypt", "sci.electronics", "sci.med", "sci.space",
    "soc.religion.christian", "talk.politics.guns", "talk.politics.mideast",
    "talk.politics.misc", "talk.religion.misc",
]


# ==============================================================================
# Page 1: Setup — trait definition
# ==============================================================================

def setup_page() -> None:
    st.header("Reward Traits Setup")
    st.caption(
        "Define the traits used to evaluate model responses. "
        "Do this before starting to grade."
    )

    _traits_editor()

    st.divider()
    _, col_btn = st.columns([3, 1])
    with col_btn:
        if st.button("Confirm Traits", type="primary", use_container_width=True):
            st.success("Traits saved. Navigate to Grade to begin.")


# ==============================================================================
# Page 2: Prompt — dataset browser + prompt editing
# ==============================================================================

def prompt_page() -> None:
    st.header("Dataset & Prompt")
    st.caption("Load a newsgroup category, browse posts, then set the article and your instruction.")

    _dataset_loader()
    _dataset_picker()

    # Seed the instruction from the saved preset the first time each session
    if not st.session_state.get("task_input", "").strip():
        preset = _load_preset_instruction()
        if preset:
            st.session_state["task_input"] = preset

    with st.form("prompt_form"):
        st.subheader("Article")
        article_val = st.text_area(
            "Article",
            value=st.session_state.get("article_input", ""),
            height=200,
            placeholder="Paste an article, or use 'Use as Article →' above to fill from the dataset.",
            label_visibility="collapsed",
        )

        preset_saved = bool(_load_preset_instruction())
        st.subheader("Instruction" + ("  ✓ saved" if preset_saved else ""))
        task_val = st.text_area(
            "Instruction",
            value=st.session_state.get("task_input", ""),
            height=100,
            placeholder="e.g. Write a response to this article, Rewrite this in a neutral tone…",
            label_visibility="collapsed",
        )

        col_set, col_clear = st.columns([3, 1])
        with col_set:
            submitted = st.form_submit_button("Set Prompt", type="primary", use_container_width=True)
        with col_clear:
            cleared = st.form_submit_button("Clear", use_container_width=True)

    if submitted:
        st.session_state["article_input"] = article_val
        st.session_state["task_input"] = task_val
        if task_val.strip():
            _save_preset_instruction(task_val)
        st.rerun()
    if cleared:
        st.session_state["article_input"] = ""
        st.session_state["task_input"] = ""
        st.rerun()

    article_set = st.session_state.get("article_input", "").strip()
    task_set    = st.session_state.get("task_input", "").strip()
    if article_set and task_set:
        st.success("Article and instruction set — go to **Grade** to begin.")
    elif article_set:
        st.caption("Article set — add an instruction and click **Set Prompt**.")
    else:
        st.caption("Fill in the fields above and click **Set Prompt**.")


# ==============================================================================
# Page 3: Grade — response input + reward scoring
# ==============================================================================

def score_page(adapter, log_path: str) -> None:
    st.header("Grade")

    article = st.session_state.get("article_input", "").strip()
    task    = st.session_state.get("task_input", "").strip()
    combined_prompt = "\n\n".join(filter(None, [article, task]))

    if article or task:
        with st.expander("Active Prompt", expanded=False):
            if article:
                st.markdown("**Article**")
                st.markdown(article)
            if article and task:
                st.divider()
            if task:
                st.markdown("**Instruction**")
                st.markdown(task)
    else:
        st.info("No prompt set. Go to **Prompt** page to choose one.")

    col_n, col_gen, col_clear = st.columns([2, 2, 1])
    with col_n:
        num = st.radio(
            "Responses",
            options=[2, 4],
            index=0 if state.get_num_responses() == 2 else 1,
            format_func=lambda n: f"{n} responses",
            horizontal=True,
        )
        if num != state.get_num_responses():
            state.set_num_responses(num)
            st.rerun()
    with col_gen:
        if st.button("Generate Responses", type="primary", use_container_width=True):
            responses = adapter.generate_responses(combined_prompt, num)
            for i, text in enumerate(responses):
                st.session_state[f"response_input_{i}"] = text
                st.session_state[f"saved_{i}"] = False
            st.rerun()
    with col_clear:
        if st.button("Clear All", use_container_width=True):
            state.clear()
            st.rerun()

    labels = ["A", "B", "C", "D"]
    summary: dict[str, float] = {}

    for i in range(num):
        label = labels[i]
        st.divider()
        st.subheader(f"Response {label}")

        response_key = f"response_input_{i}"
        saved_key    = f"saved_{i}"

        st.text_area(
            f"Response {label}",
            height=180,
            key=response_key,
            placeholder=f"Paste model response {label} here…",
            label_visibility="collapsed",
        )

        response_text = st.session_state.get(response_key, "").strip()
        if not response_text:
            st.caption(f"Paste response {label} above to score it.")
            continue

        scored_traits, scalar_reward = _scoring_panel(adapter, suffix=f"_{i}")
        if scored_traits is None:
            continue

        summary[label] = scalar_reward

        if st.session_state.get(saved_key, False):
            st.success(f"Response {label} saved to training log.")
        else:
            if st.button(f"Save Response {label}", key=f"save_btn_{i}", type="primary"):
                entry = InteractionLog(
                    timestamp=datetime.now().isoformat(timespec="seconds"),
                    prompt=combined_prompt,
                    response=response_text,
                    traits=[t.to_dict() for t in scored_traits],
                    scalar_reward=scalar_reward,
                )
                try:
                    save_interaction(entry, log_path=log_path)
                    st.session_state[saved_key] = True
                    state.mark_saved()
                    state.append_history(entry.to_dict())
                    st.rerun()
                except Exception as e:
                    st.error(f"Failed to save: {e}")

    if len(summary) > 1:
        st.divider()
        st.subheader("Score Summary")
        cols = st.columns(len(summary))
        for j, (lbl, reward) in enumerate(summary.items()):
            with cols[j]:
                st.metric(label=f"Response {lbl}", value=f"{reward:.3f}")


# ==============================================================================
# Page 3: History — master-detail
# ==============================================================================

def history_page() -> None:
    st.header("Training History")

    entries = list(reversed(state.get_history()))

    if not entries:
        st.info("No saved interactions yet. Grade and save some responses first.")
        return

    col_list, col_detail = st.columns([1, 2])

    with col_list:
        st.subheader("Sessions")
        for i, entry in enumerate(entries):
            reward = entry.get("scalar_reward", 0)
            reward_str = f"{reward:.2f}" if isinstance(reward, (int, float)) else str(reward)
            ts = entry.get("timestamp", "")[:16]
            label = f"[{reward_str}]  {ts}"
            if st.button(label, key=f"hist_btn_{i}", use_container_width=True):
                state.set_selected_history(i)

    with col_detail:
        selected = state.get_selected_history()
        if selected is None:
            st.caption("Select a session from the list to view details.")
        else:
            entry = entries[selected]
            st.subheader(entry.get("timestamp", "")[:16])

            st.markdown(f"**Prompt:** {entry.get('prompt', '')}")
            st.divider()

            st.markdown("**Response:**")
            st.markdown(entry.get("response", ""))
            st.divider()

            reward = entry.get("scalar_reward", 0)
            traits_data = entry.get("traits", [])

            metric_cols = st.columns(len(traits_data) + 1) if traits_data else st.columns(1)
            for j, t in enumerate(traits_data):
                score = t.get("score", 0)
                contrib = t.get("contribution", 0)
                with metric_cols[j]:
                    st.metric(
                        label=t.get("name"),
                        value=f"{score:+d}" if isinstance(score, int) else f"{score:+.1f}",
                        delta=f"x{t.get('weight')} = {contrib:+.3f}",
                    )
            with metric_cols[-1]:
                st.metric(
                    label="Scalar Reward",
                    value=f"{reward:+.3f}" if isinstance(reward, float) else str(reward),
                )


# ==============================================================================
# Page 4: Analytics — tabbed reward visualisation
# ==============================================================================

def analytics_page() -> None:
    st.header("Reward Analytics")

    entries = state.get_history()
    if not entries:
        st.info("No saved interactions yet. Grade and save some responses to see analytics.")
        return

    tab_trend, tab_traits, tab_stats = st.tabs(
        ["Reward Trend", "Trait Breakdown", "Statistics"]
    )

    df = pd.DataFrame([
        {
            "index": i,
            "timestamp": e.get("timestamp", "")[:16],
            "scalar_reward": e.get("scalar_reward", 0),
        }
        for i, e in enumerate(entries)
    ])

    with tab_trend:
        st.subheader("Scalar Reward Over Time")
        st.line_chart(df.set_index("timestamp")["scalar_reward"])

    with tab_traits:
        st.subheader("Average Score per Trait")
        trait_scores: dict[str, list] = {}
        for e in entries:
            for t in e.get("traits", []):
                name = t.get("name", "unknown")
                trait_scores.setdefault(name, []).append(t.get("score", 0))

        if trait_scores:
            avg_df = pd.DataFrame({
                "trait": list(trait_scores.keys()),
                "avg_score": [sum(v) / len(v) for v in trait_scores.values()],
            }).set_index("trait")
            st.bar_chart(avg_df)
        else:
            st.caption("No trait data available.")

    with tab_stats:
        st.subheader("Summary Statistics")
        rewards = [e.get("scalar_reward", 0) for e in entries]
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Sessions", len(entries))
        with col2:
            st.metric("Avg Reward", f"{sum(rewards) / len(rewards):.3f}")
        with col3:
            st.metric("Max Reward", f"{max(rewards):.3f}")
        with col4:
            st.metric("Min Reward", f"{min(rewards):.3f}")


# ==============================================================================
# Internal panel helpers
# ==============================================================================

def _traits_editor() -> None:
    st.subheader("Reward Traits")

    traits = state.get_traits()

    with st.expander("Add a trait", expanded=False):
        with st.form("add_trait_form", clear_on_submit=True):
            new_name   = st.text_input("Trait name")
            new_desc   = st.text_input("Description")
            new_weight = st.number_input("Weight", min_value=0.0, step=0.1, value=0.5)
            add_submitted = st.form_submit_button("Add trait", type="primary", use_container_width=True)
        if add_submitted:
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


def _scoring_panel(adapter, suffix: str = "") -> tuple:
    traits = state.get_traits()
    if not traits:
        st.caption("No traits defined. Go to Setup to add traits first.")
        return None, None

    scores = {}
    for trait in traits:
        scores[trait.name] = st.slider(
            f"{trait.name}  (weight {trait.weight})",
            min_value=-5, max_value=5, value=0,
            key=f"score_{trait.name}{suffix}",
        )

    try:
        scored_traits, scalar_reward = adapter.compute_reward(scores, traits)
    except Exception as e:
        st.error(f"Reward computation failed: {e}")
        return None, None

    cols = st.columns(len(traits) + 1)
    for j, st_trait in enumerate(scored_traits):
        with cols[j]:
            st.metric(
                label=st_trait.name,
                value=f"{st_trait.score:+d}",
                delta=f"x{st_trait.weight} = {st_trait.contribution:+.3f}",
            )
    with cols[-1]:
        st.metric(label="Scalar reward", value=f"{scalar_reward:+.3f}")

    return scored_traits, scalar_reward


def _save_panel(scored_traits, scalar_reward, prompt: str, response: str, log_path: str) -> None:
    if scored_traits is None:
        return

    st.subheader("Log to JSONL → GRPO Training Data")

    if state.is_saved():
        st.success("Interaction saved to training log.")
        return

    if st.button("Save interaction", type="primary"):
        entry = InteractionLog(
            timestamp=datetime.now().isoformat(timespec="seconds"),
            prompt=prompt,
            response=response,
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


def _dataset_loader() -> None:
    """Expander for selecting and loading a newsgroup category into session state."""
    loaded_cat = state.get_dataset_category()
    label = f"Load Dataset — {loaded_cat} ({len(state.get_dataset_posts())} posts)" \
            if loaded_cat else "Load Dataset"

    with st.expander(label, expanded=not loaded_cat):
        selected_cat = st.selectbox(
            "Category",
            options=_ALL_CATEGORIES,
            index=_ALL_CATEGORIES.index(loaded_cat) if loaded_cat in _ALL_CATEGORIES else 0,
            key="dataset_cat_select",
        )
        col_min, col_max = st.columns(2)
        with col_min:
            min_chars = st.slider(
                "Min length (chars)",
                min_value=50, max_value=1000, value=200, step=50,
                key="dataset_min_chars",
            )
        with col_max:
            max_chars = st.slider(
                "Max length (chars)",
                min_value=500, max_value=5000, value=1500, step=250,
                key="dataset_max_chars",
            )
        if st.button("Load", type="primary", use_container_width=True):
            with st.spinner(f"Loading {selected_cat}..."):
                raw = fetch_20newsgroups(
                    subset="all",
                    categories=[selected_cat],
                    remove=("headers", "footers", "quotes"),
                    shuffle=False,
                )
                posts = [
                    _truncate_to_boundary(c, max_chars)
                    for p in raw.data
                    if len(c := clean_text(p)) >= min_chars
                ]
            state.set_dataset_category(selected_cat)
            state.set_dataset_posts(posts)
            state.set_dataset_index(0)
            st.rerun()


def _dataset_picker() -> None:
    """Navigator for browsing loaded posts and sending one to the prompt field."""
    posts = state.get_dataset_posts()
    if not posts:
        return

    total = len(posts)
    idx = state.get_dataset_index()
    current_post = posts[idx]

    st.subheader("Choose a Post")
    st.caption(
        f"Post {idx + 1} / {total}  —  {state.get_dataset_category()}  —  {len(current_post):,} chars"
    )

    nav_prev, nav_num, nav_next, nav_rand = st.columns([1, 2, 1, 1])
    with nav_prev:
        if st.button("← Prev", use_container_width=True, disabled=idx == 0):
            state.set_dataset_index(idx - 1)
            st.rerun()
    with nav_num:
        jump = st.number_input(
            "Go to post", min_value=1, max_value=total, value=idx + 1,
            step=1, label_visibility="collapsed",
        )
        if int(jump) - 1 != idx:
            state.set_dataset_index(int(jump) - 1)
            st.rerun()
    with nav_next:
        if st.button("Next →", use_container_width=True, disabled=idx == total - 1):
            state.set_dataset_index(idx + 1)
            st.rerun()
    with nav_rand:
        if st.button("Random", use_container_width=True):
            state.set_dataset_index(random.randrange(total))
            st.rerun()

    st.text_area(
        "preview", value=current_post, height=130,
        disabled=True, label_visibility="collapsed",
    )

    if st.button("Use as Article →", type="primary", use_container_width=True):
        st.session_state["article_input"] = current_post
        st.rerun()

    st.divider()


def _inject_column_resizer() -> None:
    _components.html("""
    <script>
    (function() {
        function init() {
            var doc = window.parent.document;
            var blocks = doc.querySelectorAll('[data-testid="stHorizontalBlock"]');
            var block = blocks[0];
            if (!block) { setTimeout(init, 200); return; }
            var cols = block.querySelectorAll(':scope > [data-testid="column"]');
            if (cols.length < 2) { setTimeout(init, 200); return; }
            if (block.dataset.resizer) return;
            block.dataset.resizer = '1';

            var left = cols[0], right = cols[1];
            block.style.setProperty('display', 'flex', 'important');
            left.style.setProperty('flex', '1 1 50%', 'important');
            left.style.setProperty('min-width', '20%', 'important');
            left.style.setProperty('overflow', 'auto', 'important');
            right.style.setProperty('flex', '1 1 50%', 'important');
            right.style.setProperty('min-width', '20%', 'important');
            right.style.setProperty('overflow', 'auto', 'important');

            var handle = doc.createElement('div');
            handle.style.cssText = 'width:8px;cursor:col-resize;flex-shrink:0;display:flex;align-items:center;justify-content:center;';
            var pip = doc.createElement('div');
            pip.style.cssText = 'width:4px;height:48px;border-radius:2px;background:#d0d0d0;transition:background 0.15s;';
            handle.appendChild(pip);
            block.insertBefore(handle, right);

            var dragging = false, startX, startW;

            handle.addEventListener('mouseenter', function() { pip.style.background = '#4a90d9'; });
            handle.addEventListener('mouseleave', function() { if (!dragging) pip.style.background = '#d0d0d0'; });

            handle.addEventListener('mousedown', function(e) {
                dragging = true;
                startX = e.clientX;
                startW = left.getBoundingClientRect().width;
                doc.body.style.cursor = 'col-resize';
                doc.body.style.userSelect = 'none';
                e.preventDefault();
            });

            doc.addEventListener('mousemove', function(e) {
                if (!dragging) return;
                var total = block.getBoundingClientRect().width - 8;
                var newW = Math.min(Math.max(startW + e.clientX - startX, total * 0.2), total * 0.8);
                left.style.setProperty('flex', '0 0 ' + newW + 'px', 'important');
                right.style.setProperty('flex', '0 0 ' + (total - newW) + 'px', 'important');
            });

            doc.addEventListener('mouseup', function() {
                if (!dragging) return;
                dragging = false;
                pip.style.background = '#d0d0d0';
                doc.body.style.cursor = '';
                doc.body.style.userSelect = '';
            });
        }
        init();
    })();
    </script>
    """, height=0)


# ==============================================================================
# Page 6: Train — GRPO fine-tuning on saved interactions
# ==============================================================================

def train_page(log_path: str, grpo_output_dir: str = "grpo_output") -> None:
    from pathlib import Path as _Path
    from grpo_adapter import (
        GRPOSession, load_scored_prompts, load_session_records,
        zip_adapter_dir, merge_and_export,
    )

    active_session = state.get_active_session()

    st.header("GRPO Training")
    st.caption(
        f"Session: **{active_session}** — fine-tune Qwen2.5 on saved interactions using GRPO. "
        "Grade responses on the **Grade** page first to build up training data."
    )

    # ── Data summary ────────────────────────────────────────────────────────────
    prompt_reward_map = load_scored_prompts(log_path)
    n_prompts = len(prompt_reward_map)
    session   = state.get_grpo_session()

    col_a, col_b, col_c = st.columns(3)
    with col_a:
        st.metric("Scored Prompts", n_prompts)
    with col_b:
        rewards = list(prompt_reward_map.values())
        st.metric("Avg Reward", f"{sum(rewards)/len(rewards):.3f}" if rewards else "—")
    with col_c:
        st.metric("Last Run", session.status if session else "none")

    if n_prompts == 0:
        st.info("No training data yet. Go to **Grade**, score some responses, and save them.")
        _past_sessions_panel(load_session_records, log_path)
        return

    st.divider()

    # ── Config ──────────────────────────────────────────────────────────────────
    st.subheader("Training Config")
    col_ep, col_note = st.columns([1, 3])
    with col_ep:
        num_epochs = st.number_input("Epochs", min_value=1, max_value=10, value=1, step=1)
    with col_note:
        st.caption(
            f"Model: Qwen/Qwen2.5-1.5B-Instruct + LoRA  \n"
            f"Adapters → `{grpo_output_dir}/`"
        )

    # ── Controls ────────────────────────────────────────────────────────────────
    training_active = session is not None and not session.done

    col_start, col_refresh = st.columns([1, 1])
    with col_start:
        if st.button(
            "Start Training", type="primary",
            use_container_width=True, disabled=training_active,
        ):
            new_session = GRPOSession(
                log_path=log_path,
                num_epochs=int(num_epochs),
                adapter_dir=grpo_output_dir,
            )
            new_session.start()
            state.set_grpo_session(new_session)
            st.rerun()
    with col_refresh:
        if training_active and st.button("Refresh Log", use_container_width=True):
            st.rerun()

    # ── Progress log ────────────────────────────────────────────────────────────
    session = state.get_grpo_session()
    if session is not None:
        st.divider()
        st.subheader("Training Log")

        _STATUS_COLOR = {
            "idle": "gray", "loading": "blue", "training": "orange",
            "done": "green", "error": "red",
        }
        color = _STATUS_COLOR.get(session.status, "gray")
        st.markdown(f"**Status:** :{color}[{session.status.upper()}]")

        if session.error:
            st.error(session.error)
        if session.output_dir:
            st.success(f"LoRA adapters saved to `{session.output_dir}/`")

        log_text = "\n".join(session.log) if session.log else "Waiting…"
        st.code(log_text, language=None)

        if training_active:
            st.caption("Click **Refresh Log** to see latest progress.")

    # ── Export ──────────────────────────────────────────────────────────────────
    adapter_dir = (
        (session.output_dir if session and session.output_dir else None)
        or (grpo_output_dir if _Path(grpo_output_dir).exists() else None)
    )

    if adapter_dir and _Path(adapter_dir).exists():
        st.divider()
        st.subheader("Export")

        tab_zip, tab_merge = st.tabs(["Download Adapters (ZIP)", "Merge & Export Full Model"])

        with tab_zip:
            st.caption(
                "Download the LoRA adapter files. "
                "Load with `PeftModel.from_pretrained(base_model, adapter_dir)`."
            )
            if st.button("Prepare ZIP", use_container_width=True):
                with st.spinner("Zipping adapter files…"):
                    zip_bytes = zip_adapter_dir(adapter_dir)
                st.download_button(
                    label="Download adapters.zip",
                    data=zip_bytes,
                    file_name="grpo_adapters.zip",
                    mime="application/zip",
                    use_container_width=True,
                )

        with tab_merge:
            st.caption(
                "Merge LoRA weights into the base model and save a standalone model — "
                "no PEFT needed to load the result."
            )
            merge_out = st.text_input("Output directory", value="grpo_merged_model", key="merge_out_dir")
            if st.button("Merge & Save", type="primary", use_container_width=True):
                with st.spinner(f"Merging LoRA into base weights and saving to {merge_out}/…"):
                    try:
                        merge_and_export(adapter_dir, merge_out)
                        st.success(f"Merged model saved to `{merge_out}/`")
                    except Exception as e:
                        st.error(f"Merge failed: {e}")

    # ── Past sessions ────────────────────────────────────────────────────────────
    _past_sessions_panel(load_session_records, log_path)


def _past_sessions_panel(load_fn, log_path: str) -> None:
    records = load_fn(log_path)
    if not records:
        return

    st.divider()
    st.subheader("Past Sessions")

    for rec in records:
        ts      = rec.get("timestamp", "")[:16]
        status  = rec.get("status", "?")
        epochs  = rec.get("num_epochs", "?")
        n_p     = rec.get("n_prompts", "?")
        out_dir = rec.get("output_dir") or "—"
        label   = f"{ts}  ·  {status.upper()}  ·  {n_p} prompts  ·  {epochs} epoch(s)"

        with st.expander(label, expanded=False):
            st.caption(f"Adapters: `{out_dir}`")
            if rec.get("error"):
                st.error(rec["error"])
            log_lines = rec.get("log", [])
            if log_lines:
                st.code("\n".join(log_lines), language=None)
