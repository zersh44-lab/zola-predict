# Zola Predict - Project Status

## Project Overview

**Zola Predict** is a local-first football research and prediction assistant running on Windows 11. It provides evidence-based match research, statistical market forecasting across six independent models, historical data backtesting, and local dashboard/Telegram delivery.

---

## 5-Day Roadmap & Implementation State

| Day | Focus Area | Status |
|---|---|---|
| **Day 1** | **Project foundation and football data collection** | **COMPLETED** |
| **Day 2** | **Historical data foundation & AI research integration** | **COMPLETED** (Channel/social monitoring deferred) |
| **Day 3** | **Six statistical prediction models & quality checks** | **COMPLETED** |
| **Day 4** | **Local dashboard and Telegram report delivery** | **COMPLETED** |
| Day 5 | Performance tracking, testing, and corrections | Planned |

---

## Completed Features & Architecture

### 1. Football Data Sources & Provenance
- **API-Football (v3)**: Primary live data provider for fixtures, leagues, and live results.
- **football-data.org (v4)**: Alternate/secondary live data adapter.
- **Historical FootballData CSVs (`jokecamp/FootballData`)**:
  - Validated CSV importer (`HistoricalCsvImporter`) supporting FootballData / Football-Data.co.uk formats.
  - Strict provenance recording in `historical_data_imports` (SHA-256 file hash, season, source repository, row count).
  - Matches tagged with provider `football-data-csv` to cleanly separate historical backtesting data from live API data.

### 2. Six Independent Statistical Prediction Models
1. **Poisson Distribution Model (`poisson`)**:
   - Calculates league attack and defense relative strength coefficients.
   - Computes independent Poisson goal expectancies ($\lambda_{\text{home}}, \lambda_{\text{away}}$).
2. **Dixon-Coles Model (`dixon_coles`)**:
   - Adjusts standard Poisson for low-score correlation parameter ($\rho \approx -0.11$) for 0-0, 1-0, 0-1, and 1-1 scorelines.
   - Applies exponential time-decay weighting ($\xi$) prioritizing recent match form.
3. **Elo Dynamic Power Rating Model (`elo`)**:
   - Replays chronological match results with goal-margin sensitivity.
   - Adjusts for home ground advantage (+65 Elo rating points) and models draw probabilities using logistic distribution.
4. **Recent Form & Momentum Model (`form_momentum`)**:
   - Evaluates weighted points-per-game (PPG), goal differentials, and clean-sheet streaks over the last 5-10 matches.
   - Incorporates home/away performance splits.
5. **Head-to-Head Historical Matchup Model (`head_to_head`)**:
   - Analyzes historical clashes between the two competing clubs with recency decay.
   - Gracefully handles sparse direct history via Bayesian shrinkage toward league priors.
6. **Consensus Ensemble Meta-Model (`ensemble`)**:
   - Combines individual model distributions dynamically weighted by each model's confidence score.
   - Measures inter-model variance (entropy / disagreement).
   - Generates transparent data quality notes and modulates consensus confidence.

### 3. Six Core Prediction Markets
For every fixture and model, probabilities and top picks are calculated for:
1. `1X2`: Home Win / Draw / Away Win
2. `BTTS`: Both Teams to Score (Yes / No)
3. `OU25`: Over / Under 2.5 Total Goals
4. `OU15`: Over / Under 1.5 Total Goals
5. `DOUBLE_CHANCE`: 1X, X2, 12
6. `DNB`: Draw No Bet (Home DNB, Away DNB)

### 4. Data Quality Checks & Uncertainty Safeguards
- Sparse sample size detection (< 5 matches per team).
- Model divergence alerts: flags `HIGH_DIVERGENCE` when models disagree on outcome direction.
- Confidence levels: `HIGH`, `MEDIUM`, `LOW`, or `UNAVAILABLE`.
- Never presents unsupported certainty; data quality notes are stored alongside every prediction in SQLite.

### 5. Prediction Evaluation & Accuracy Metrics
- `evaluate-predictions`: Compares pending predictions with actual match results when matches finish.
- Evaluates outcome correctness across all 6 markets.
- Tracks accuracy leaderboard and Brier scores in `accuracy_metrics` table.

### 6. AI Research Provider Interface
- Supports Google Gemini, Groq, OpenRouter, and local Ollama (`http://localhost:11434`, zero API key required).
- Includes deterministic `StatisticalFallbackProvider` that synthesizes research summaries without needing external AI keys.

### 7. Deferred Source Monitoring
- Generic source registry is preserved.
- The 8 user-entered Telegram sources remain saved in `monitored_sources` and remain `PAUSED`.
- Live channel polling, YouTube scraping, and social transcripts remain deferred to future milestones.

### 8. Local Web Dashboard (`zola/dashboard/`)
- **Zero Cloud / Zero Paid Dependencies**: Multi-threaded local HTTP server (`zola/dashboard/server.py`) using Python standard library `http.server`.
- **Pure Responsive Frontend**: Standalone single-page app (`index.html`) using HTML5, CSS3, and modern JavaScript. Requires no internet connection or external CDN to render.
- **REST API Endpoints**:
  - `GET /api/status`: System health, DB stats, credential states, cache & rate limit info.
  - `GET /api/competitions`: Stored football leagues.
  - `GET /api/fixtures`: Filterable fixture list (by status, competition, or limit).
  - `GET /api/predict?id=<id>`: Full 6-model prediction matrix + consensus probabilities.
  - `GET /api/research?id=<id>`: Tactical match research synthesis.
  - `GET /api/metrics`: Live accuracy metrics leaderboard.
  - `POST /api/evaluate`: Scores pending predictions against finished match results.
  - `GET /api/telegram/info`: Bot username, status, and discovered chat IDs.
  - `POST /api/telegram/send`: Direct match card delivery or test greeting to Telegram.
- **One-Click Windows Launcher**: `launch_dashboard.bat` and `python main.py dashboard`.

### 9. Telegram Bot Delivery (`zola/telegram/`)
- **Direct Match Card Delivery**: `TelegramReportSender` delivers rich prediction cards to private or group chats via `@ZolaPredict_bot`.
- **Chat Discovery**: `python main.py telegram-chats` lists chat IDs of users who interacted with the bot.
- **Delivery Commands**:
  - `python main.py send-report --id <fixture_id> [--chat-id <id>]`
  - `python main.py test-telegram [--chat-id <id>]`
- **Entity Parse Safety**: Auto-retries as plain text if Markdown entity formatting encounters unsupported characters in club names.
- **Credential Protection**: Token is strictly read from `.env` and never leaked or logged.

---

## Database Schema Entities

| Table | Purpose |
|---|---|
| `competitions` | Football leagues and divisions |
| `teams` | Football clubs |
| `fixtures` | Scheduled, in-play, and finished match fixtures |
| `match_results` | Full-time, half-time, extra-time, and penalty scores |
| `research_sources` | Live and historical data source catalog |
| `monitored_sources` | User-selected channels/pages (Telegram sources saved & paused) |
| `source_items` | Posts or articles discovered by future collectors |
| `historical_data_imports` | Provenance records (SHA-256 hash, filename, season, row count) |
| `predictions` | Model predictions per market with confidence, features, and evaluation |
| `accuracy_metrics` | Accuracy leaderboard and performance metrics per model and market |
| `api_collection_logs` | Audit trail of external live API requests |
| `request_cache` | SHA-256 persistent request cache with TTL to protect free quota |

---

## Test Execution Summary

```powershell
pytest -v
```

```
============================= 50 passed in 3.99s ==============================
```
- **Total Tests**: 50
- **Failures / Errors**: 0
- **External Network Calls During Tests**: 0 (100% mocked / in-memory SQLite)

---

## Environment Variables Status

| Variable | Status | Role |
|---|---|---|
| `FOOTBALL_API_KEY` | `[PRESENT]` | Live API-Football key for match sync |
| `GEMINI_API_KEY` | `[PRESENT]` | Optional Google Gemini research key |
| `TELEGRAM_BOT_TOKEN`| `[PRESENT]` | Telegram bot report delivery (`@ZolaPredict_bot`) |
| `TELEGRAM_CHAT_ID` | `[MISSING]` | Optional default chat ID for 1-click delivery |
| `GROQ_API_KEY` | `[MISSING]` | Optional Groq inference key |
| `OPENROUTER_API_KEY`| `[MISSING]` | Optional OpenRouter key |
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Local offline LLM endpoint |

---

## Next Steps (Day 5: Performance Tracking, Testing, and Corrections)

1. Long-term accuracy tracking and Brier score calibration analysis across larger historical fixture samples.
2. End-to-end integration stress tests across multiple leagues (Premier League, La Liga, Serie A).
3. Minor UI refinements and final Windows 11 packaging.
