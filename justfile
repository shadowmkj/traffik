# Traffik Task Runner
# List all available recipes
default:
    @just --list

# ==============================================================================
# Testing & Verification
# ==============================================================================

# Run the complete test suite with uv
test:
    uv run pytest tests/

# Run tests with verbose output
test-verbose:
    uv run pytest -vv tests/

# Run a specific test file or target (e.g., just test-file tests/test_gate.py)
test-file path="tests/test_gate.py":
    uv run pytest {{path}}

# ==============================================================================
# Video Processing & Analytics
# ==============================================================================

# Process video headlessly at maximum speed
process video="clips/clip_11.mp4" config="configs/default.toml":
    uv run traffik process {{video}} -c {{config}}

# Stream video playback with real-time OpenCV window
stream video="clips/clip_11.mp4" config="configs/default.toml":
    uv run traffik stream {{video}} -c {{config}}

# Process video with license plate OCR extraction enabled
ocr video="clips/clip_1.mp4" config="configs/default.toml":
    uv run traffik process {{video}} --ocr -c {{config}}

# Launch interactive 4-point virtual gate calibration tool
setup-gate video="clips/clip_4.mp4":
    uv run traffik setup-gate {{video}}

# Launch interactive 4-point speed calibration ROI tool
setup-speed-roi video="clips/clip_4.mp4":
    uv run traffik setup-speed-roi {{video}}

# ==============================================================================
# Maintenance & Environment
# ==============================================================================

# Sync virtualenv dependencies with uv
sync:
    uv sync

# Clean temporary caches and bytecode
clean:
    rm -rf .pytest_cache __pycache__ traffik/**/__pycache__ tests/__pycache__
