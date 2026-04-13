"""
Toy test for Milestone 1 (Config + Trait Loader).

Tests that traits.json loads correctly and returns valid trait dicts.

Run with: uv run python tests/test_milestones/toy_test_m1.py
"""

import sys
from pathlib import Path

# Walk up from this file until we find the app/ folder (the reward_system root)
project_root = Path(__file__).parent
while project_root.parent != project_root:
    if (project_root / "app").exists():
        break
    project_root = project_root.parent

sys.path.insert(0, str(project_root))

from app.trait_loader import load_traits


def print_section(title):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")


def test_load_traits():
    """Test that traits.json loads and returns a list of dicts."""
    print_section("TEST 1: Load traits from config")

    config_path = project_root / "config" / "traits.json"
    print(f"  Loading from: {config_path}")

    try:
        traits = load_traits(str(config_path))
        print(f"  ✓ Loaded {len(traits)} traits")

        assert isinstance(traits, list), "load_traits should return a list"
        assert len(traits) > 0, "Trait list should not be empty"

        print(f"  ✓ Return type is list")
        return traits
    except Exception as e:
        print(f"  ✗ Failed to load traits: {e}")
        raise


def test_trait_fields(traits):
    """Test that each trait has name, description, and weight."""
    print_section("TEST 2: Validate trait fields")

    required = {"name", "description", "weight"}

    try:
        for trait in traits:
            missing = required - trait.keys()
            assert not missing, f"Trait '{trait.get('name', '?')}' missing fields: {missing}"

        print(f"  ✓ All traits have required fields: {required}")
    except Exception as e:
        print(f"  ✗ Field validation failed: {e}")
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

        print(f"  ✓ All field types are correct")
    except Exception as e:
        print(f"  ✗ Type validation failed: {e}")
        raise


def test_print_traits(traits):
    """Print each trait so the user can visually confirm the values."""
    print_section("TEST 4: Print loaded traits")

    for trait in traits:
        print(f"  - {trait['name']:<16} weight={trait['weight']}  |  {trait['description']}")

    print(f"\n  ✓ Traits look correct above")


def test_missing_file():
    """Test that a missing config raises FileNotFoundError."""
    print_section("TEST 5: Missing config file raises error")

    try:
        load_traits("config/does_not_exist.json")
        print(f"  ✗ Should have raised FileNotFoundError")
        raise AssertionError("Expected FileNotFoundError but got none")
    except FileNotFoundError:
        print(f"  ✓ Correctly raised FileNotFoundError for missing file")


def main():
    print("\n" + "="*60)
    print("  REWARD SYSTEM - MILESTONE 1 TEST SUITE")
    print("  Testing Config + Trait Loader")
    print("="*60)

    try:
        traits = test_load_traits()
        test_trait_fields(traits)
        test_trait_types(traits)
        test_print_traits(traits)
        test_missing_file()

        print("\n" + "="*60)
        print("  ✓ ALL MILESTONE 1 TESTS PASSED")
        print("="*60 + "\n")

    except Exception as e:
        print("\n" + "="*60)
        print(f"  ✗ TEST FAILED: {e}")
        print("="*60 + "\n")
        raise


if __name__ == "__main__":
    main()
