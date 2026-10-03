# "Seeing Next Month, Not Just Last Month" — Predictive Dashboard

## Microsoft Hackathon Project

### Overview
A predictive dashboard solution providing forward-looking insights, subscription dynamics, and churn analytics for SaaS businesses.

### Project Structure
```text
Microsoft hackathon project/
│
├── data/
│   ├── raw/               # Original, unmodified datasets (e.g., SaaS Subscription & Churn Analytics Dataset)
│   └── processed/         # Cleaned and transformed datasets ready for modeling/analysis
│
├── src/
│   ├── __init__.py        # Package initializer
│   ├── pipeline.py        # Data pipeline and workflow functions
│   └── config.py          # Configuration and directory path constants
│
├── notebooks/             # Exploratory Data Analysis (EDA) and experimental notebooks
│
├── tests/                 # Unit and integration test suites
│
├── outputs/               # Generated reports, figures, charts, and export artifacts
│
├── models/                # Saved model weights, checkpoints, and serialization files
│
├── requirements.txt       # Project dependencies
├── README.md              # Project documentation
├── .gitignore             # Git ignore patterns
└── main.py                # Main project entry point
```

### Setup & Usage
1. Upload the raw dataset to `data/raw/`.
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the project:
   ```bash
   python main.py
   ```
