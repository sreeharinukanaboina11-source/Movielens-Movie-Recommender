# CSE303 Machine Learning: Week 10 Recommendation Systems

This repository contains the completed **CSE303 Machine Learning Week 10 Recommendation Systems Mini Project** on the official GroupLens MovieLens dataset.

## Project Structure
```
.
├── labs/
│   └── week10_recsys/
│       └── team_recsys/
│           ├── notebook.ipynb      # Complete executed notebook with all outputs and figures
│           ├── README.md           # Full project documentation, results tables, and logs
│           ├── requirements.txt    # Project dependencies
│           ├── cv_bullet.txt       # Measured 2-line CV bullet
│           ├── app.py              # Interactive recommender demo (CLI & Streamlit)
│           └── data/
│               └── README.md       # Dataset source, schema, and download instructions
├── Week10_Project_Starter_Template.ipynb # Preserved original starter template
├── Week10_Recommendation_Systems_Student_Lab.ipynb # Preserved lab reference notebook
├── Week10_Recsys_Project_Menu3.pdf # Authoritative project specification
└── .gitignore
```

## Quick Links
- **Completed Deliverable Notebook:** [labs/week10_recsys/team_recsys/notebook.ipynb](labs/week10_recsys/team_recsys/notebook.ipynb)
- **Full Project Documentation & Results Table:** [labs/week10_recsys/team_recsys/README.md](labs/week10_recsys/team_recsys/README.md)
- **Dataset Instructions:** [labs/week10_recsys/team_recsys/data/README.md](labs/week10_recsys/team_recsys/data/README.md)

## Quick Start
```bash
# 1. Install dependencies
pip install -r labs/week10_recsys/team_recsys/requirements.txt

# 2. Run the interactive CLI recommender
python labs/week10_recsys/team_recsys/app.py --user 1 --n 10 --model hybrid

# 3. Or launch the Streamlit web app
python -m streamlit run labs/week10_recsys/team_recsys/app.py
```
