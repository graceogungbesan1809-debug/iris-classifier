# Iris Classifier (Decision Tree)

## Overview
End‑to‑end ML example from Digital Marketing Mastery Module → builds a decision‑tree classifier on the classic Iris dataset using scikit‑learn.

## Quick start
```bash
git clone https://github.com/graceogungbesan1809-debug/iris-classifier.git
cd iris-classifier
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
python src/train.py

Project structure
iris-classifier/
├── data/ # (empty – Iris is loaded from scikit‑learn)
├── notebooks/
│ └── iris_model.ipynb # walk‑through notebook
├── src/
│ └── train.py # reproducible CLI script
├── tests/
│ └── test_train.py # basic pytest
├── outputs/ # created automatically (model & figures)
├── .gitignore
├── LICENSE
├── README.md
└── requirements.txt


License
 Use the MIT licence unless you have reasons to choose another.



Optional tests in tests/ – e.g., assert that accuracy ≥ 0.9.

