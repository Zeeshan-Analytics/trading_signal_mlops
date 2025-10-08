# 📈 Trading Signal Generator - MLOps Project

An end-to-end MLOps project that generates intraday trading signals for US equities using Neural Networks. This project demonstrates best practices in ML engineering including experiment tracking, data versioning, automated pipelines, and deployment.

## 🎯 Project Overview

- **Market**: US Equities (S&P 500 stocks)
- **Timeframe**: Hourly (Intraday)
- **Signals**: Multi-class classification
  - 📈 Strong Buy
  - ⬆️ Buy
  - ➡️ Hold
  - ⬇️ Sell
  - 📉 Strong Sell
- **Retraining**: Weekly (automated)
- **Data Source**: yfinance


## 🎨 Architecture
Data Fetching (yfinance)
↓
Feature Engineering (Technical Indicators)
↓
Model Training (TensorFlow + Wandb)
↓
Model Registry (DVC + Wandb)
↓
API Service (FastAPI → Render)
↓
Web Dashboard (Streamlit → Streamlit Cloud)
↓
Monitoring & Retraining (GitHub Actions)


## 🛠️ Tech Stack
| Component | Technology |
|-----------|-----------|
| **ML Framework** | TensorFlow/Keras |
| **Data Processing** | Pandas, NumPy, Scikit-learn |
| **Data Source** | yfinance |
| **Technical Analysis** | ta, pandas-ta |
| **Experiment Tracking** | Weights & Biases (Wandb) |
| **Data Versioning** | DVC (Data Version Control) |
| **API** | FastAPI |
| **Web Dashboard** | Streamlit |
| **Testing** | Pytest |
| **CI/CD** | GitHub Actions |
| **Deployment** | Render (API), Streamlit Cloud (Dashboard) |

## 📁 Project Structure
trading-signal-mlops/
├── .github/
│   └── workflows/        # GitHub Actions CI/CD pipelines
├── api/                  # FastAPI application
├── app/                  # Streamlit dashboard
├── configs/              # Configuration files
│   ├── config.yaml       # Main project configuration
│   └── dvc.yaml          # DVC pipeline configuration
├── data/                 # Data directories (DVC tracked)
│   ├── raw/              # Raw data from yfinance
│   ├── processed/        # Processed and cleaned data
│   └── features/         # Engineered features
├── logs/                 # Application and training logs
├── models/
│   └── saved_models/     # Trained models (DVC tracked)
├── notebooks/            # Jupyter notebooks for EDA
├── src/                  # Source code
│   ├── data/             # Data fetching and processing
│   ├── features/         # Feature engineering
│   ├── models/           # Model architecture and training
│   └── utils/            # Utility functions
├── tests/                # Unit and integration tests
├── .env                  # Environment variables (not in git)
├── .gitignore
├── requirements.txt      # Python dependencies
└── README.md