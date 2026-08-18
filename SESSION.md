# Incidental Findings and Deferrals

- `main.py` uses `output.mp4` as `TARGET_VIDEO_PATH`, whereas `README.md` references `output_counted.mp4`.
- NumPy 2.0 cross-product monkey patch is placed directly in `main.py` rather than a separate compatibility utility.
