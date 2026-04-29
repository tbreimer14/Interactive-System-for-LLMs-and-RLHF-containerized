"""Milestone 5: Training loop

Stage A — imports and wiring (no model download, no GPU)
Stage B — full training run (requires GPU + connected UI scorer)

Run Stage A:
    uv run python tests/test_milestones/toy_test_m5.py

Run Stage B (real training):
    uv run python tests/test_milestones/toy_test_m5.py --full
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))


def test_imports():
    from grpo_system.train import train, get_scores_from_ui
    from grpo_system.config import GRPOConfig
    print("  imports: OK")


def test_get_scores_from_ui_raises():
    from grpo_system.train import get_scores_from_ui
    try:
        get_scores_from_ui("prompt", "response")
        assert False, "should have raised NotImplementedError"
    except NotImplementedError:
        pass
    print("  get_scores_from_ui raises NotImplementedError: OK")


def test_train_accepts_custom_scorer():
    """train() must accept get_scores_fn without calling it during setup."""
    from grpo_system.train import train
    from grpo_system.config import GRPOConfig
    import inspect

    sig = inspect.signature(train)
    assert "get_scores_fn" in sig.parameters, "train() must accept get_scores_fn param"
    assert "config" in sig.parameters, "train() must accept config param"
    print("  train() signature correct: OK")


def test_full_training_run():
    """Full training step — requires GPU and a UI scorer."""
    from grpo_system.train import train
    from grpo_system.config import GRPOConfig
    import tempfile

    calls = []

    def mock_scorer(prompt, response):
        calls.append(1)
        return {"interactivity": 4, "warmth": 3, "directness": 5}

    with tempfile.TemporaryDirectory() as tmpdir:
        cfg = GRPOConfig(
            batch_size=1,
            num_generations=2,
            num_train_epochs=1,
            max_completion_length=16,  # shortest possible completion
            output_dir=tmpdir,
        )
        # pass max_articles=1 so only one prompt is loaded — single training step
        import grpo_system.data as _data
        _orig = _data.load_articles
        _data.load_articles = lambda **kw: _orig(max_articles=1)
        try:
            train(config=cfg, get_scores_fn=mock_scorer)
        finally:
            _data.load_articles = _orig

        assert len(calls) > 0, "scorer must have been called during training"
        print(f"  scorer called {len(calls)} times: OK")

        adapter_config = os.path.join(tmpdir, "adapter_config.json")
        assert os.path.exists(adapter_config), "adapter_config.json must be saved"
        print(f"  adapter_config.json saved: OK")

        tokenizer_file = os.path.join(tmpdir, "tokenizer.json")
        assert os.path.exists(tokenizer_file), "tokenizer.json must be saved"
        print(f"  tokenizer.json saved: OK")


if __name__ == "__main__":
    full = "--full" in sys.argv

    print("=== Milestone 5: Training Loop ===")
    test_imports()
    test_get_scores_from_ui_raises()
    test_train_accepts_custom_scorer()

    if full:
        print("\n  Running full training step (requires GPU)...")
        test_full_training_run()
    else:
        print("\n  [Stage B skipped] Run with --full to do a real training step")

    print("\nMilestone 5 tests passed.")
