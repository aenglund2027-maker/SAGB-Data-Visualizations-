"""
Step 1: download (if missing), load, and clean ATP main-tour matches 2014-2026.

Output: data/processed/atp_matches_clean.csv
Run from the project root:  python3 run_01_load_clean.py
"""
from src import config
from src.data_loading import clean_matches, describe_clean_data, download_raw_files, load_raw_matches

if __name__ == "__main__":
    download_raw_files()
    raw = load_raw_matches()
    clean = clean_matches(raw)
    describe_clean_data(clean)

    config.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    clean.to_csv(config.CLEAN_MATCHES_PATH, index=False)
    print(f"\nSaved -> {config.CLEAN_MATCHES_PATH}")
