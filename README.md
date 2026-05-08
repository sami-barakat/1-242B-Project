# Sentiment-Powered LSTM with Multi-Armed Bandit Trading

**INDENG 1/242B — Machine Learning and Data Analytics II**
**University of California, Berkeley — Spring 2026**

**Team:** Sami Barakat, Ryan Chekkouri, Haley Choe, Justin Choi, Aanvi Koolwal

## Overview

An adaptive trading system for U.S. energy equities that combines sentiment analysis, sequence modeling, and adaptive learning to dynamically allocate across competing trading strategies.

**Pipeline:**
1. Daily news + social media text from NewsAPI, GDELT, and Reddit → **FinBERT** → ticker-level sentiment scores
2. Sentiment features merged with 17 engineered price features → **2-layer LSTM** → next-day market regime probabilities (trending / mean-reverting / high-volatility) for 14 energy tickers
3. LSTM regime probabilities → **Exp3 multi-armed bandit** (one per regime) → daily allocation across 7 trading strategies, weights updated from realized returns

Evaluated on observed 2026 market data. Full methodology and results are in the project report.

## Repository Structure

| File | Purpose |
|------|---------|
| [`DataPull.ipynb`](DataPull.ipynb) | Pulls daily OHLCV for 14 energy tickers from Yahoo Finance (Jan 2020 – Apr 2026) |
| [`SentimentAnalysis.ipynb`](SentimentAnalysis.ipynb) | Collects text from NewsAPI / Reddit (PRAW) / GDELT, scores with FinBERT, produces daily sentiment features |
| [`LSTM_Regime_Model.ipynb`](LSTM_Regime_Model.ipynb) | Trains the 2-layer LSTM regime classifier; outputs regime probability vectors per (ticker, date) |
| [`MAB_Final_RealValues.ipynb`](MAB_Final_RealValues.ipynb) | Final MAB evaluation on observed 2026 test data — defines the 7 trading strategies, the EXP3 bandit, and runs the full evaluation end-to-end |
| [`242B Final Project.ipynb`](242B%20Final%20Project.ipynb) | End-to-end data cleaning notebook |

**Cached intermediate outputs** (committed so the pipeline can be reproduced without re-running expensive API calls):

| File | Produced by | Consumed by |
|------|-------------|-------------|
| `raw_corpus_cache.csv` | `SentimentAnalysis.ipynb` (raw scraped text) | `SentimentAnalysis.ipynb` (FinBERT scoring step) |
| `sentiment_features.csv` | `SentimentAnalysis.ipynb` (FinBERT-scored, daily-aggregated) | `LSTM_Regime_Model.ipynb` |
| `sentiment_scaler_stats.csv` | `SentimentAnalysis.ipynb` | `LSTM_Regime_Model.ipynb` |
| `test_regime_probabilities.csv` | `LSTM_Regime_Model.ipynb` | MAB notebooks |
| `Train_Dataset_No_Sentiment.csv`, `Val_Dataset_No_Sentiment.csv`, `Test_Dataset_No_Sentiment.csv` | `DataPull.ipynb` | MAB notebooks |

## How to Run

```bash
git clone https://github.com/sami-barakat/1-242B-Project.git
cd 1-242B-Project
pip install -r requirements.txt
jupyter notebook
```

Then open any notebook and **Run All**. Every notebook reads and writes CSVs in the current directory — no Google Drive setup or manual file uploads required.

**To reproduce the final results:** open `MAB_Final_RealValues.ipynb` and Run All. It is self-contained — it reads the cached `Test_Dataset_No_Sentiment.csv` and `test_regime_probabilities.csv` (both committed to this repo) and produces the final bandit evaluation.

**To inspect how the inputs were generated**, the dev pipeline runs in this order:
1. `DataPull.ipynb` — pulls daily prices from Yahoo Finance
2. `SentimentAnalysis.ipynb` — produces sentiment features from NewsAPI / Reddit / GDELT
3. `242B Final Project.ipynb` — cleans and splits the dataset into train / val / test
4. `LSTM_Regime_Model.ipynb` — trains the regime classifier and exports regime probabilities

All intermediate outputs from these steps are already cached in the repo, so re-running them is optional.

## API Credentials

`SentimentAnalysis.ipynb` requires three credentials **only if you re-pull data** from scratch:

- `NEWS_API_KEY` — free account at [newsapi.org](https://newsapi.org/)
- `REDDIT_CLIENT_ID` and `REDDIT_CLIENT_SECRET` — script-type app at [reddit.com/prefs/apps](https://www.reddit.com/prefs/apps)

Set them as environment variables (or via `google.colab.userdata` in Colab). **No keys are needed to reproduce the final results** — call `run_pipeline(load_cached_corpus="raw_corpus_cache.csv")` to skip data collection and use the cached corpus.
