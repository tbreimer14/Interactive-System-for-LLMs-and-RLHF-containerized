"""
Toy test for Milestone 2 (Trait Manager + Config).

Tests that config/traits.json loads correctly via trait_manager.load_traits().

Run with: uv run python tests/test_milestones/toy_test_m2.py
"""

import sys
from pathlib import Path

# Walk up from this file until we find the app/ folder (the user_interface root)
project_root = Path(__file__).parent
while project_root.parent != project_root:
    if (project_root / "app").exists():
        break
    project_root = project_root.parent

sys.path.insert(0, str(project_root))

from app.trait_manager import load_traits

CONFIG_PATH = project_root / "config" / "traits.json"


def print_section(title):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")


def test_load_traits():
    """Test that traits.json loads and returns a non-empty list."""
    print_section("TEST 1: Load traits from config")

    print(f"  Loading from: {CONFIG_PATH}")

    try:
        traits = load_traits(str(CONFIG_PATH))
        assert isinstance(traits, list), "load_traits should return a list"
        assert len(traits) > 0, "Trait list should not be empty"
        print(f"  [PASS] Loaded {len(traits)} traits")
        return traits
    except Exception as e:
        print(f"  [FAIL] {e}")
        raise


def test_trait_fields(traits):
    """Test that each trait has name, description, and weight."""
    print_section("TEST 2: Validate trait fields")

    required = {"name", "description", "weight"}

    try:
        for trait in traits:
            missing = required - trait.keys()
            assert not missing, f"Trait '{trait.get('name', '?')}' missing fields: {missing}"
        print(f"  [PASS] All traits have required fields: {required}")
    except Exception as e:
        print(f"  [FAIL] {e}")
        raise


def test_trait_types(traits):
    """Test that field types are correct."""
    print_section("TEST 3: Validate field types")

    try:
        for trait in traits:
            assert isinstance(trait["name"], str), \
                f"'name' should be str, got {type(trait['name'])}"
            assert isinstance(trait["description"], str), \
                f"'description' should be str, got {type(trait['description'])}"
            assert isinstance(trait["weight"], (int, float)), \
                f"'weight' should be a number, got {type(trait['weight'])}"
        print(f"  [PASS] All field types are correct")
    except Exception as e:
        print(f"  [FAIL] {e}")
        raise


def test_weights_positive(traits):
    """Test that all weights are positive numbers."""
    print_section("TEST 4: Weights are positive")

    try:
        for trait in traits:
            assert trait["weight"] > 0, \
                f"Trait '{trait['name']}' has non-positive weight: {trait['weight']}"
        print(f"  [PASS] All weights are positive")
    except Exception as e:
        print(f"  [FAIL] {e}")
        raise


def test_missing_file():
    """Test that a missing config raises FileNotFoundError."""
    print_section("TEST 5: Missing config file raises error")

    try:
        load_traits("config/does_not_exist.json")
        print(f"  [FAIL] Should have raised FileNotFoundError")
        raise AssertionError("Expected FileNotFoundError but got none")
    except FileNotFoundError:
        print(f"  [PASS] Correctly raised FileNotFoundError for missing file")


def test_print_traits(traits):
    """Print each trait so the user can visually confirm the values."""
    print_section("TEST 6: Visual output")

    for trait in traits:
        print(f"  - {trait['name']:<16} weight={trait['weight']}  |  {trait['description']}")

    print(f"\n  [PASS] Traits look correct above")


def main():
    print("\n" + "=" * 60)
    print("  USER INTERFACE - MILESTONE 2 TEST SUITE")
    print("  Testing Trait Manager + Config")
    print("=" * 60)

    try:
        traits = test_load_traits()
        test_trait_fields(traits)
        test_trait_types(traits)
        test_weights_positive(traits)
        test_missing_file()
        test_print_traits(traits)

        print("\n" + "=" * 60)
        print("  ALL MILESTONE 2 TESTS PASSED")
        print("=" * 60 + "\n")

    except Exception as e:
        print("\n" + "=" * 60)
        print(f"  TEST FAILED: {e}")
        print("=" * 60 + "\n")
        raise


if __name__ == "__main__":
    main()
