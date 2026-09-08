# Incidental Findings and Deferrals

- `main.py` uses `output.mp4` as `TARGET_VIDEO_PATH`, whereas `README.md` references `output_counted.mp4`.
- Supervision 0.29+ emits a `FutureWarning` indicating `ByteTrack` class will be migrated in supervision v0.30.0; track upstream API update when upgrading.
