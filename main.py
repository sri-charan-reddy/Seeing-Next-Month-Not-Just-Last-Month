"""
Main entry point for "Seeing Next Month, Not Just Last Month" - Predictive Dashboard.
"""

from src.config import BASE_DIR, RAW_DATA_DIR, PROCESSED_DATA_DIR


def main():
    print("Project initialized: 'Seeing Next Month, Not Just Last Month' — Predictive Dashboard")
    print(f"Base Directory: {BASE_DIR}")
    print(f"Raw Data Directory: {RAW_DATA_DIR}")
    print(f"Processed Data Directory: {PROCESSED_DATA_DIR}")


if __name__ == "__main__":
    main()
