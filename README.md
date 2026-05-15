# AutoGluon on Ames Housing: Can a 5-Minute Model Beat a Weekend of Feature Engineering?

A complete AutoML case study using [AutoGluon](https://auto.gluon.ai/) on the Ames Housing dataset.  
Built for Extra HW2 - AutoGluon Exploration.

---

## Project Summary

This project evaluates AutoGluon as a rapid AutoML solution for tabular regression, using the classic Ames Housing dataset (predict `SalePrice` from 79 features). We compare:

- **AutoGluon `medium_quality`** - 2-minute fast baseline
- **AutoGluon `best_quality`** - full ensemble with multi-layer stacking
- **Scikit-learn manual pipeline** - GBM with median imputation + one-hot encoding

Key finding: AutoGluon's best_quality model reduced RMSE by ~24% vs. a manually-tuned scikit-learn baseline, training in under 10 minutes with no feature engineering.

---

## Repository Structure

```
.
├── data               
    └── AmesHousing.csv     # Data
├── notebook.ipynb          # Full AutoGluon pipeline + SHAP explainability
├── article.md              # Medium-style technical writeup (~750 words)
├── requirements.txt        # Python dependencies
├── README.md               # This file
└── figures/
    ├── saleprice_distribution.png
    ├── leaderboard.png
    ├── feature_importance.png
    ├── shap_summary.png
    ├── shap_waterfall.png
    ├── model_comparison.png
    └── residuals.png
```

---

## Setup & Reproduction

### Option A: Local Environment

```bash
# Clone the repo
git clone https://github.com/definitely-el/autogluon-house-prices.git
cd autogluon-house-prices

# Create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Launch Jupyter
jupyter notebook notebook.ipynb
```
---

## Key Results

| Method | RMSE (Test) |
|--------|-------------|
| Scikit-learn GBM (manual) | ~$26,000 |
| AutoGluon `medium_quality` | ~$23,500 |
| AutoGluon `best_quality` | ~$19,800 |

AutoGluon trained 30+ model variants and assembled a 3-layer stacked ensemble. The best single base model (LightGBM) was wrapped with SHAP for individual prediction explanations.

---

## Tools & Libraries

- [AutoGluon 1.x](https://auto.gluon.ai/) - AutoML framework
- [SHAP](https://shap.readthedocs.io/) - model explainability
- [scikit-learn](https://scikit-learn.org/) - baseline pipeline
- [pandas](https://pandas.pydata.org/), [numpy](https://numpy.org/) - data handling
- [matplotlib](https://matplotlib.org/), [seaborn](https://seaborn.pydata.org/) - visualization

---

## Dataset

**Ames Housing Dataset** - 2,930 residential sales in Ames, Iowa (2006–2010)  
82 explanatory features including lot size, neighborhood, quality ratings, and structural attributes  
Source: https://www.kaggle.com/datasets/prevek18/ames-housing-dataset

---

## License

MIT
