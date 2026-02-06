#!/usr/bin/env python3
"""
Launch script for Semantic Server with Gemma 3 and Prisoner's Dilemma experiment.

This script:
1. Checks if Gemma 3 model is downloaded
2. Downloads model if needed (with progress)
3. Clears old cache files
4. Kills any existing semantic/streamlit processes
5. Launches semantic server
6. Launches Streamlit coordination UI
7. Displays instructions for the Prisoner's Dilemma experiment
"""

import os
import sys
import time
import shutil
import signal
import subprocess
from pathlib import Path

# Configuration
MODEL_ID = "mlx-community/gemma-3-12b-it-4bit"
MODEL_SHORT_NAME = "gemma-3-12b-it-4bit"
SERVER_PORT = 8000
STREAMLIT_PORT = 8501
CACHE_DIR = Path.home() / ".semantic" / "caches"
HF_CACHE_DIR = Path.home() / ".cache" / "huggingface" / "hub"

# Colors for terminal output
class Colors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'

def print_header(msg):
    print(f"\n{Colors.HEADER}{Colors.BOLD}{'='*60}{Colors.ENDC}")
    print(f"{Colors.HEADER}{Colors.BOLD}{msg}{Colors.ENDC}")
    print(f"{Colors.HEADER}{Colors.BOLD}{'='*60}{Colors.ENDC}\n")

def print_step(step_num, msg):
    print(f"{Colors.CYAN}[Step {step_num}]{Colors.ENDC} {msg}")

def print_success(msg):
    print(f"{Colors.GREEN}✓ {msg}{Colors.ENDC}")

def print_warning(msg):
    print(f"{Colors.YELLOW}⚠ {msg}{Colors.ENDC}")

def print_error(msg):
    print(f"{Colors.RED}✗ {msg}{Colors.ENDC}")

def print_info(msg):
    print(f"{Colors.BLUE}ℹ {msg}{Colors.ENDC}")


def check_model_available():
    """Check if the Gemma 3 model is downloaded."""
    print_step(1, "Checking if Gemma 3 model is available...")

    # Check HuggingFace cache for the model
    model_cache_pattern = MODEL_ID.replace("/", "--")

    if HF_CACHE_DIR.exists():
        for item in HF_CACHE_DIR.iterdir():
            if model_cache_pattern in item.name and item.is_dir():
                # Check if it has actual model files
                snapshots = item / "snapshots"
                if snapshots.exists():
                    for snapshot in snapshots.iterdir():
                        if snapshot.is_dir():
                            files = list(snapshot.glob("*.safetensors"))
                            if files:
                                print_success(f"Model found in cache: {item.name}")
                                return True

    print_warning("Model not found in local cache")
    return False


def download_model():
    """Download the Gemma 3 model using huggingface_hub."""
    print_step(2, f"Downloading model: {MODEL_ID}")
    print_info("This may take a while (model is ~7GB)...")

    try:
        # Try using huggingface_hub
        from huggingface_hub import snapshot_download, HfApi

        # Check if user is logged in
        api = HfApi()
        try:
            api.whoami()
            print_info("HuggingFace authentication detected")
        except Exception:
            print_warning("Not logged in to HuggingFace (may be needed for some models)")

        # Download with progress
        print_info("Starting download...")
        snapshot_download(
            repo_id=MODEL_ID,
            repo_type="model",
            resume_download=True,
        )
        print_success("Model downloaded successfully!")
        return True

    except ImportError:
        print_error("huggingface_hub not installed")
        show_setup_instructions()
        return False
    except Exception as e:
        print_error(f"Download failed: {e}")
        show_setup_instructions()
        return False


def show_setup_instructions():
    """Show instructions for setting up HuggingFace access."""
    print_header("HuggingFace Setup Instructions")

    print(f"""
{Colors.YELLOW}To download the Gemma 3 model, you need:{Colors.ENDC}

1. {Colors.BOLD}Install huggingface_hub:{Colors.ENDC}
   pip install huggingface_hub

2. {Colors.BOLD}Create a HuggingFace account:{Colors.ENDC}
   https://huggingface.co/join

3. {Colors.BOLD}Accept the Gemma license:{Colors.ENDC}
   https://huggingface.co/{MODEL_ID}
   (Click "Agree and access repository")

4. {Colors.BOLD}Login to HuggingFace CLI:{Colors.ENDC}
   huggingface-cli login
   (Enter your access token from https://huggingface.co/settings/tokens)

5. {Colors.BOLD}Re-run this script:{Colors.ENDC}
   python launch_experiment.py

{Colors.CYAN}Alternative: Manual download{Colors.ENDC}
   - Go to https://huggingface.co/{MODEL_ID}
   - Download all files to ~/.cache/huggingface/hub/models--{MODEL_ID.replace('/', '--')}
""")


def clear_cache_files():
    """Clear old semantic cache files."""
    print_step(3, "Clearing old cache files...")

    if CACHE_DIR.exists():
        cache_count = len(list(CACHE_DIR.glob("*.safetensors")))
        if cache_count > 0:
            print_info(f"Found {cache_count} cache files in {CACHE_DIR}")
            try:
                for f in CACHE_DIR.glob("*.safetensors"):
                    f.unlink()
                print_success(f"Cleared {cache_count} cache files")
            except Exception as e:
                print_warning(f"Could not clear all caches: {e}")
        else:
            print_info("No cache files to clear")
    else:
        print_info("Cache directory does not exist yet")


def kill_processes_by_pattern(pattern, description):
    """Kill processes matching a pattern."""
    try:
        # Find PIDs
        result = subprocess.run(
            ["pgrep", "-f", pattern],
            capture_output=True,
            text=True
        )
        pids = result.stdout.strip().split('\n')
        pids = [p for p in pids if p]

        if pids:
            print_info(f"Found {len(pids)} {description} process(es)")
            for pid in pids:
                try:
                    os.kill(int(pid), signal.SIGKILL)
                except (ProcessLookupError, ValueError):
                    pass
            print_success(f"Killed {description} processes")
            time.sleep(1)
        else:
            print_info(f"No {description} processes running")
    except Exception as e:
        print_warning(f"Could not check for {description} processes: {e}")


def kill_existing_processes():
    """Kill any existing semantic server and streamlit processes."""
    print_step(4, "Stopping existing processes...")

    # Kill semantic server processes
    kill_processes_by_pattern("semantic serve", "semantic server")

    # Kill by port
    try:
        result = subprocess.run(
            ["lsof", f"-ti:{SERVER_PORT}"],
            capture_output=True,
            text=True
        )
        if result.stdout.strip():
            pids = result.stdout.strip().split('\n')
            for pid in pids:
                try:
                    os.kill(int(pid), signal.SIGKILL)
                except (ProcessLookupError, ValueError):
                    pass
            print_success(f"Cleared port {SERVER_PORT}")
    except Exception:
        pass

    # Kill streamlit processes for coordination
    kill_processes_by_pattern("streamlit.*coordination", "streamlit coordination")

    # Also check streamlit port
    try:
        result = subprocess.run(
            ["lsof", f"-ti:{STREAMLIT_PORT}"],
            capture_output=True,
            text=True
        )
        if result.stdout.strip():
            pids = result.stdout.strip().split('\n')
            for pid in pids:
                try:
                    os.kill(int(pid), signal.SIGKILL)
                except (ProcessLookupError, ValueError):
                    pass
            print_success(f"Cleared port {STREAMLIT_PORT}")
    except Exception:
        pass

    time.sleep(2)


def launch_semantic_server():
    """Launch the semantic server with Gemma 3."""
    print_step(5, f"Launching Semantic Server on port {SERVER_PORT}...")

    log_file = Path("/tmp/semantic_server.log")

    cmd = [
        "semantic", "serve",
        "--model", MODEL_ID,
        "--port", str(SERVER_PORT)
    ]

    with open(log_file, 'w') as f:
        process = subprocess.Popen(
            cmd,
            stdout=f,
            stderr=subprocess.STDOUT,
            start_new_session=True
        )

    print_info(f"Server PID: {process.pid}")
    print_info(f"Log file: {log_file}")

    # Wait for server to be ready
    print_info("Waiting for server to start...")
    max_wait = 60
    for i in range(max_wait):
        try:
            import urllib.request
            with urllib.request.urlopen(f"http://localhost:{SERVER_PORT}/health", timeout=2) as resp:
                if resp.status == 200:
                    print_success("Semantic server is ready!")
                    return True
        except Exception:
            pass

        # Check if process died
        if process.poll() is not None:
            print_error("Server process died unexpectedly")
            print_info("Check log file for details:")
            os.system(f"tail -20 {log_file}")
            return False

        time.sleep(1)
        if i % 10 == 9:
            print_info(f"Still waiting... ({i+1}s)")

    print_error(f"Server did not start within {max_wait} seconds")
    return False


def launch_streamlit():
    """Launch the Streamlit coordination UI."""
    print_step(6, f"Launching Streamlit UI on port {STREAMLIT_PORT}...")

    # Find the streamlit app
    script_dir = Path(__file__).parent
    streamlit_app = script_dir / "apps" / "streamlit_coordination.py"

    if not streamlit_app.exists():
        # Try alternative locations
        alt_paths = [
            script_dir / "streamlit_coordination.py",
            script_dir / "src" / "semantic" / "apps" / "streamlit_coordination.py",
        ]
        for alt in alt_paths:
            if alt.exists():
                streamlit_app = alt
                break

    if not streamlit_app.exists():
        print_warning(f"Streamlit app not found at {streamlit_app}")
        print_info("You can manually run: streamlit run apps/streamlit_coordination.py")
        return False

    log_file = Path("/tmp/streamlit.log")

    cmd = [
        "streamlit", "run",
        str(streamlit_app),
        "--server.port", str(STREAMLIT_PORT),
        "--server.headless", "true"
    ]

    with open(log_file, 'w') as f:
        process = subprocess.Popen(
            cmd,
            stdout=f,
            stderr=subprocess.STDOUT,
            start_new_session=True,
            cwd=str(script_dir)
        )

    print_info(f"Streamlit PID: {process.pid}")
    time.sleep(3)

    if process.poll() is None:
        print_success("Streamlit UI is running!")
        return True
    else:
        print_warning("Streamlit may not have started correctly")
        return False


def show_experiment_instructions():
    """Display instructions for running the Prisoner's Dilemma experiment."""
    print_header("Prisoner's Dilemma Experiment")

    print(f"""
{Colors.GREEN}{Colors.BOLD}Services Running:{Colors.ENDC}
  • Semantic Server: http://localhost:{SERVER_PORT}
  • Streamlit UI:    http://localhost:{STREAMLIT_PORT}

{Colors.CYAN}{Colors.BOLD}Option 1: Use the Streamlit UI{Colors.ENDC}
  1. Open your browser to: {Colors.BOLD}http://localhost:{STREAMLIT_PORT}{Colors.ENDC}
  2. Configure two agents:

     {Colors.YELLOW}Agent 1 (Prisoner A):{Colors.ENDC}
     Name: Prisoner A
     System Prompt:
     You are Prisoner A in a Prisoner's Dilemma game.
     RULES:
     - Both COOPERATE: 1 year each
     - Both DEFECT: 3 years each
     - One DEFECTS, other COOPERATES: Defector free, cooperator 5 years
     State clearly: "I choose to COOPERATE" or "I choose to DEFECT"

     {Colors.YELLOW}Agent 2 (Prisoner B):{Colors.ENDC}
     Name: Prisoner B
     System Prompt: (same as above, but "You are Prisoner B")

  3. Set directive: "Round 1: Make your choice for the Prisoner's Dilemma."
  4. Click "Create Session" and then run turns

{Colors.CYAN}{Colors.BOLD}Option 2: Run the Python Script{Colors.ENDC}
  python /tmp/claude/prisoners_dilemma.py

{Colors.CYAN}{Colors.BOLD}Option 3: Use curl directly{Colors.ENDC}
  # Create session
  curl -X POST http://localhost:{SERVER_PORT}/v1/coordination/sessions \\
    -H "Content-Type: application/json" \\
    -d '{{"agents": [...], "directive": "..."}}'

  # Execute turns
  curl -X POST http://localhost:{SERVER_PORT}/v1/coordination/sessions/{{session_id}}/turn

{Colors.YELLOW}{Colors.BOLD}What to Observe:{Colors.ENDC}
  • Do agents remember previous round outcomes?
  • Do they adapt their strategy based on opponent's behavior?
  • Is there any identity confusion?
  • How does timing vary between cold and warm cache turns?

{Colors.GREEN}{Colors.BOLD}Logs:{Colors.ENDC}
  • Server log: tail -f /tmp/semantic_server.log
  • Streamlit log: tail -f /tmp/streamlit.log
  • Cache stats: grep "CACHE" /tmp/semantic_server.log

{Colors.RED}{Colors.BOLD}To stop all services:{Colors.ENDC}
  pkill -f "semantic serve"
  pkill -f "streamlit"
""")


def main():
    print_header("Semantic Server Launch Script")
    print(f"Model: {MODEL_ID}")
    print(f"Server Port: {SERVER_PORT}")
    print(f"Streamlit Port: {STREAMLIT_PORT}")

    # Step 1 & 2: Check/download model
    if not check_model_available():
        if not download_model():
            print_error("Cannot proceed without model")
            sys.exit(1)

    # Step 3: Clear cache
    clear_cache_files()

    # Step 4: Kill existing processes
    kill_existing_processes()

    # Step 5: Launch semantic server
    if not launch_semantic_server():
        print_error("Failed to start semantic server")
        sys.exit(1)

    # Step 6: Launch streamlit
    launch_streamlit()

    # Step 7: Show instructions
    show_experiment_instructions()

    print_success("Setup complete! Follow the instructions above to run experiments.")


if __name__ == "__main__":
    main()
