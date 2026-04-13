"""
Milestone 5 Tests: Build Index Script & Chat CLI

Tests:
1. test_build_index_script_execution: Verify build_index.py runs and builds index
2. test_build_index_success_output: Check success message in output
3. test_cli_loads_index: Verify chat_cli.py loads index at startup
4. test_cli_query_loop_simulation: Verify CLI processes queries correctly
5. test_cli_exit_command: Verify exit command terminates CLI
"""

import subprocess
import sys
from pathlib import Path
import shutil
import tempfile


def find_app_module():
    """Find the app module by traversing up from test file."""
    current = Path(__file__).parent
    for _ in range(5):
        if (current / "app").exists():
            sys.path.insert(0, str(current))
            return current
        current = current.parent
    raise RuntimeError("Could not find app module")


root_dir = find_app_module()
import os
os.chdir(root_dir)

from app.pipeline import RAGPipeline


print("=" * 60)
print("Milestone 5: Build Index Script & Chat CLI Tests")
print("=" * 60)


def test_build_index_script_execution():
    """Test 1: Verify build_index.py runs and creates index."""
    print("\n" + "=" * 60)
    print("TEST 1: Build Index Script Execution")
    print("=" * 60)
    
    # Create temp dir for test index
    with tempfile.TemporaryDirectory() as tmpdir:
        out_dir = Path(tmpdir) / "test_index"
        
        # Run build_index.py
        cmd = [
            sys.executable, 
            "scripts/build_index.py",
            "tests/data/raw",
            str(out_dir)
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        # Check exit code
        if result.returncode != 0:
            print(f"[ERROR] Script failed with exit code {result.returncode}")
            print(f"STDOUT:\n{result.stdout}")
            print(f"STDERR:\n{result.stderr}")
            return False
        
        # Check index was created
        if not out_dir.exists():
            print(f"[ERROR] Index directory not created at {out_dir}")
            return False
        
        # Check index files exist
        required_files = ["dataset_info.json", "state.json"]
        for file in required_files:
            if not (out_dir / file).exists():
                print(f"[ERROR] Required file {file} not found in index directory")
                return False
        
        print(f"[OK] Index successfully built at {out_dir}")
        return True


def test_build_index_success_output():
    """Test 2: Verify success message in build_index.py output."""
    print("\n" + "=" * 60)
    print("TEST 2: Build Index Success Output")
    print("=" * 60)
    
    with tempfile.TemporaryDirectory() as tmpdir:
        out_dir = Path(tmpdir) / "test_index"
        
        cmd = [
            sys.executable,
            "scripts/build_index.py",
            "tests/data/raw",
            str(out_dir)
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        # Check for success message and directory in output
        if "[OK] Index built and saved to" in result.stdout:
            print(f"[OK] Success message found in output")
            return True
        else:
            print(f"[ERROR] Success message not found")
            print(f"Output:\n{result.stdout}")
            return False


def test_cli_loads_index():
    """Test 3: Verify chat_cli.py loads index at startup."""
    print("\n" + "=" * 60)
    print("TEST 3: CLI Loads Index")
    print("=" * 60)
    
    # First build an index
    with tempfile.TemporaryDirectory() as tmpdir:
        out_dir = Path(tmpdir) / "test_index"
        
        pipeline = RAGPipeline()
        pipeline.build_index(raw_data_dir="tests/data/raw", out_dir=str(out_dir))
        
        # Now test loading with CLI script (simulate with exit input)
        cmd = [sys.executable, "scripts/chat_cli.py", str(out_dir)]
        
        # Simulate pressing 'exit' immediately
        result = subprocess.run(
            cmd,
            input="exit\n",
            capture_output=True,
            text=True,
            timeout=30
        )
        
        # Check for loading message
        if "Loading RAG index" in result.stdout or "Loaded" in result.stdout:
            print(f"[OK] Index loading message found")
            return True
        else:
            print(f"[ERROR] Index loading not confirmed")
            print(f"Output:\n{result.stdout}")
            return False


def test_cli_query_loop_simulation():
    """Test 4: Verify CLI processes queries and returns results."""
    print("\n" + "=" * 60)
    print("TEST 4: CLI Query Loop Simulation")
    print("=" * 60)
    
    # Build test index
    with tempfile.TemporaryDirectory() as tmpdir:
        out_dir = Path(tmpdir) / "test_index"
        
        pipeline = RAGPipeline()
        pipeline.build_index(raw_data_dir="tests/data/raw", out_dir=str(out_dir))
        
        # Simulate user query then exit
        cmd = [sys.executable, "scripts/chat_cli.py", str(out_dir)]
        
        test_input = "What is machine learning?\nexit\n"
        
        result = subprocess.run(
            cmd,
            input=test_input,
            capture_output=True,
            text=True,
            timeout=60
        )
        
        # Check for query phrase in output
        if "Query" in result.stdout or "Retrieved" in result.stdout or "Answer" in result.stdout:
            print(f"[OK] CLI processed query and returned results")
            return True
        else:
            print(f"[ERROR] CLI did not process query correctly")
            print(f"Output:\n{result.stdout}")
            return False


def test_cli_exit_command():
    """Test 5: Verify CLI exit command terminates gracefully."""
    print("\n" + "=" * 60)
    print("TEST 5: CLI Exit Command")
    print("=" * 60)
    
    # Build test index
    with tempfile.TemporaryDirectory() as tmpdir:
        out_dir = Path(tmpdir) / "test_index"
        
        pipeline = RAGPipeline()
        pipeline.build_index(raw_data_dir="tests/data/raw", out_dir=str(out_dir))
        
        cmd = [sys.executable, "scripts/chat_cli.py", str(out_dir)]
        
        # Test 'exit' command
        result = subprocess.run(
            cmd,
            input="exit\n",
            capture_output=True,
            text=True,
            timeout=30
        )
        
        if result.returncode == 0 or "Goodbye" in result.stdout:
            print(f"[OK] CLI exited gracefully with 'exit' command")
            return True
        else:
            print(f"[ERROR] CLI did not exit gracefully")
            return False


def main():
    """Run all tests."""
    tests = [
        ("Build Index Script Execution", test_build_index_script_execution),
        ("Build Index Success Output", test_build_index_success_output),
        ("CLI Loads Index", test_cli_loads_index),
        ("CLI Query Loop Simulation", test_cli_query_loop_simulation),
        ("CLI Exit Command", test_cli_exit_command),
    ]
    
    results = []
    for name, test_fn in tests:
        try:
            passed = test_fn()
            results.append((name, passed))
        except Exception as e:
            print(f"\n[ERROR] TEST FAILED: {e}")
            import traceback
            traceback.print_exc()
            results.append((name, False))
    
    # Summary
    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)
    for name, passed in results:
        status = "[OK]" if passed else "[ERROR]"
        print(f"{status} {name}")
    
    passed_count = sum(1 for _, p in results if p)
    total_count = len(results)
    print(f"\nPassed: {passed_count}/{total_count}")
    
    if passed_count == total_count:
        print("\n[OK] All tests passed!")
        return 0
    else:
        print(f"\n[ERROR] {total_count - passed_count} test(s) failed")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
