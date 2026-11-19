# Zola Predict

**Zola Predict** is a local-first football research and match-prediction assistant designed for Windows 11. It collects football match data, stores it in an ACID-compliant local SQLite database, and runs **six independent statistical prediction models** with probability calculations, quality checks, and performance tracking.

---

## Football Data Sources

- **API-Football (v3)**: Primary live data provider for competitions, fixtures, and scores.
- **Historical FootballData CSVs (`jokecamp/FootballData`)**: Curated historical CSV datasets for backtesting and statistical baseline calculation. Provenance is tracked with SHA-256 hashes and isolated under the `football-data-csv` provider tag so it is never confused with live data.
- **Optional live provider**: `football-data.org` (v4).

---

## Six Independent Prediction Models

1. **Poisson Distribution Model (`poisson`)**: Calculates relative attack/defense strength ratings and Poisson goal expectancies ($\lambda$).
2. **Dixon-Coles Model (`dixon_coles`)**: Adjusts Poisson probabilities for low-score correlation (0-0, 1-0, 0-1, 1-1) with exponential time-decay weighting on recent matches.
3. **Elo Dynamic Power Rating Model (`elo`)**: Replays team performance with goal-margin sensitivity, home field advantage (+65 points), and logistic draw modeling.
4. **Recent Form & Momentum Model (`form_momentum`)**: Analyzes weighted points-per-game (PPG), goal differential, and home/away performance splits over the last 5-10 matches.
5. **Head-to-Head Historical Matchup Model (`head_to_head`)**: Evaluates direct historical clashes between the two clubs with recency weighting and Bayesian shrinkage.
6. **Consensus Ensemble Meta-Model (`ensemble`)**: Synthesizes all 5 models dynamically weighted by confidence scores, measures inter-model variance, and flags model divergence.

### Six Core Prediction Markets
Every model predicts probability distributions for:
- **1X2**: Home Win / Draw / Away Win
- **BTTS**: Both Teams to Score (Yes / No)
- **OU25**: Over / Under 2.5 Total Goals
- **OU15**: Over / Under 1.5 Total Goals
- **DOUBLE_CHANCE**: 1X, 12, X2
- **DNB**: Draw No Bet (Home DNB, Away DNB)

---

## Windows 11 Setup Guide

### 1. Prerequisites
- Windows 11 (64-bit)
- Python 3.11+ (or `uv` package manager)
- Windows PowerShell

### 2. Quick Setup (Automated)
Run the automated PowerShell script from the `ZolaPredict` directory:
```powershell
powershell -ExecutionPolicy Bypass -File .\setup_windows.ps1
```

### 3. Manual Step-by-Step Setup

#### Step A: Create and Activate Virtual Environment
```powershell
# Using uv (recommended on Windows):
uv venv .venv
.\.venv\Scripts\activate

# Or using standard python:
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

#### Step B: Install Dependencies
```powershell
pip install -r requirements.txt
```

#### Step C: Configure Credentials in `.env`
Copy `.env.example` to `.env`:
```powershell
Copy-Item .env.example .env
```
Open `.env` in your text editor and add your rotated credentials:
```ini
FOOTBALL_API_KEY=your_api_football_key_here
GEMINI_API_KEY=
TELEGRAM_BOT_TOKEN=

# Optional AI Research Providers (Gemini, Groq, OpenRouter, local Ollama)
GROQ_API_KEY=
OPENROUTER_API_KEY=
OLLAMA_BASE_URL=http://localhost:11434
AI_PROVIDER=auto

# Live provider configuration
FOOTBALL_API_PROVIDER=api-football
ZOLA_DB_PATH=data/zola_predict.db
ZOLA_CACHE_TTL_MINUTES=60
ZOLA_RATE_LIMIT_RPM=10
```

> [!IMPORTANT]
> Never commit `.env` or share your API keys. Zola Predict redacts all credentials in logs, CLI output, and string representations.

#### Step D: Initialize the SQLite Database
```powershell
python main.py init-db
```

---

## CLI Usage Instructions

You can run commands using `python main.py <command>` or the `.\run.bat <command>` launcher:

### 1. Check Configuration (Credential Safe)
```powershell
.\run.bat check-env
```

### 2. View Database Status
```powershell
.\run.bat status
```

### 3. Synchronize Live Competitions & Fixtures
```powershell
# Synchronize competitions (API-Football)
.\run.bat sync-competitions

# Sync live upcoming matches for today across active competitions
.\run.bat sync-fixtures --all --date today

# Sync upcoming matches for tomorrow
.\run.bat sync-fixtures --all --date tomorrow

# Collect upcoming fixtures for a specific competition (e.g. Premier League ID 39)
.\run.bat sync-fixtures --competition 39 --status SCHEDULED

# Collect finished match scores for calibration
.\run.bat sync-results --competition 39 --limit 20

# View genuine upcoming matches scheduled locally
.\run.bat list-fixtures --upcoming
```

### 4. Import Historical FootballData CSVs
Import curated datasets with SHA-256 provenance tracking:
```powershell
.\run.bat import-csv --file data\sample_historical_pl.csv --competition E0 --season 2023-2024
```

### 5. Run the Prediction Engine on a Match
Generates 1X2, BTTS, Over/Under, Double Chance, and DNB probabilities across all 6 models:
```powershell
.\run.bat predict-match --id 17
```

### 6. Batch Predict Upcoming Fixtures
```powershell
.\run.bat predict-fixtures --limit 10
```

### 7. Evaluate Past Predictions & View Accuracy Leaderboard
```powershell
# Evaluate predictions against finished match scores
.\run.bat evaluate-predictions

# View accuracy leaderboard by model and market
.\run.bat model-stats
```

### 8. AI-Assisted Football Research Report
Synthesize an evidence-based match research brief using local Ollama, Gemini, Groq, or statistical fallback:
```powershell
.\run.bat research-match --id 17
```

---

## Local Web Dashboard (Day 4)

Zola Predict includes a fast, modern, local-first web dashboard that runs entirely on your laptop without any cloud dependencies.

### Launching the Dashboard

You can start the dashboard in any of the following ways:

```powershell
# Method 1: Double-click or run launch_dashboard.bat
.\launch_dashboard.bat

# Method 2: Via main.py CLI
python main.py dashboard

# Method 3: Via run.bat with custom port
.\run.bat dashboard --port 8080
```

This will automatically open your default browser to `http://127.0.0.1:8080`.

### Dashboard Features
- **Overview & Stat Counters**: Stored competitions, fixtures, evaluated predictions, and consensus hit rate.
- **Interactive Fixtures & Match Predictor**:
  - Filter by scheduled/finished matches or search by team.
  - Interactive prediction cards with visual probability bars for all 6 markets (`1X2`, `BTTS`, `OU25`, `OU15`, `DOUBLE_CHANCE`, `DNB`).
  - Model comparison matrix comparing Poisson, Dixon-Coles, Elo, Form/Momentum, Head-to-Head, and Consensus Ensemble.
  - Data quality alerts and model divergence warnings.
- **Accuracy Leaderboard**: Evaluates models against real completed scores with 1-click re-evaluation.
- **Telegram Delivery**: Instant "Send Card to Telegram" action directly from any match prediction card.
- **Security & Audit View**: Real-time credential masking status and audit trail of local API cache hits/misses.

---

## Telegram Bot Prediction Delivery (Day 4)

Zola Predict can push rich, formatted match prediction cards directly to your private Telegram chat or group via your personal Telegram bot (`@ZolaPredict_bot`).

> [!NOTE]
> Telegram delivery pushes prediction reports *out* to you; it does *not* monitor or scrape external channels.

### Setup & Usage

1. **Configure Bot Token in `.env`**:
   Add your bot token (already configured if using the rotated token):
   ```ini
   TELEGRAM_BOT_TOKEN=your_bot_token_here
   ```

2. **Discover your Chat ID**:
   Open Telegram, search for your bot (e.g., `@ZolaPredict_bot`), and send `/start` or a greeting. Then run:
   ```powershell
   .\run.bat telegram-chats
   ```
   This will list your `Chat ID` (e.g., `123456789`).

3. **Optional: Set Default Chat ID in `.env`**:
   To avoid typing `--chat-id` on every command:
   ```ini
   TELEGRAM_CHAT_ID=123456789
   ```

4. **Send a Test Greeting**:
   ```powershell
   .\run.bat test-telegram
   ```

5. **Send a Match Prediction Card to Telegram**:
   ```powershell
   .\run.bat send-report --id 1
   ```

---

## Source Registry (Deferred Features)

The source registry tracks user-entered channels and accounts. Active polling, YouTube transcripts, and social monitoring remain deferred to future phases. Existing saved Telegram channels remain safely preserved in the database and are paused.

```powershell
.\run.bat source-list
.\run.bat source-status --id 1 --status PAUSED
```

---

## Running Automated Tests

All tests execute locally with **zero external network requests** and **no real API keys required**:

```powershell
pytest -v
```

Expected output:
```
============================= 47 passed in 4.77s ==============================
```
