# AutoGluon on Ames Housing: Can a 5-Minute Model Beat a Weekend of Feature Engineering?

A complete AutoML case study using [AutoGluon](https://auto.gluon.ai/) on the Ames Housing dataset.  
Built for Extra HW2 - AutoGluon Exploration.

---

## Project Summary

This project evaluates AutoGluon as a rapid AutoML solution for tabular regression, using the classic Ames Housing dataset (predict `SalePrice` from 79 features). We compare:

- **AutoGluon `medium_quality`** - 2-minute fast baseline
- **AutoGluon `best_quality`** - full ensemble with multi-layer stacking
- **Scikit-learn manual pipeline** - GBM with median imputation + one-hot encoding

**Key finding:** Both AutoGluon presets beat the manual scikit-learn baseline. Interestingly, `medium_quality` ($23,202 RMSE) edged out `best_quality` ($23,774 RMSE) on the test set — because AutoGluon 1.5's DyStack detected stacking overfitting and automatically disabled multi-layer stacking, causing both presets to converge on L1-only ensembles. This is documented and discussed in the notebook and article.

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

### Local Environment

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

> **Note:** `best_quality` with `time_limit=600` takes ~10 minutes on a CPU-only machine (tested on Windows, AMD64, 8 cores, no CUDA). The `feature_importance` cell takes ~4 minutes. The SHAP cell takes ~11 minutes.

---

## Environment

Tested on:
- Python 3.11.7, Windows 10
- AutoGluon 1.5.0
- PyTorch 2.9.1 (CPU only — no CUDA)
- 8-core AMD64, 16 GB RAM

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
