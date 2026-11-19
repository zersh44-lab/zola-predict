# BeSoccer-Inspired Football Match Discovery & Prioritization System
## Architecture & Implementation Plan

### 1. Existing System Assessment
- **Backend**: Python HTTP server (`zola/dashboard/server.py`) serving REST endpoints and static files from `zola/dashboard/static/`.
- **Database**: SQLite database at `data/zola_predict.db` using `zola/database/repository.py` and `models.py`.
- **Data Collection**: `zola/services/collector.py` using `zola/client/api_sports.py` with caching (`RequestCache`) and rate-limiting (`RateLimiter`).
- **Prediction Engine**: 6 statistical models + Consensus Ensemble (`zola/prediction/engine.py`, `ensemble.py`, `dixon_coles.py`).
- **Current Match Display**: A flat `<table>` inside `index.html` displaying basic match rows.

---

### 2. Architecture & Design

```
+----------------------------------------------------------------------------------------+
|                                    ZOLA PREDICT UI                                     |
|  +----------------------------------------------------------------------------------+  |
|  | Date Navigation: [<] [Nov 15] [Nov 16] [Today (Nov 17)] [Nov 18] [>] [Calendar]   |  |
|  +----------------------------------------------------------------------------------+  |
|  | Category Filter: All (166) | Live (0) | Upcoming (166) | Finished (0) | Favs (0) |  |
|  +----------------------------------------------------------------------------------+  |
|  | Search & Controls: [Search teams/league...] [Country] [Sort: Recommended]        |  |
|  +----------------------------------------------------------------------------------+  |
|  | SECTION 1: Featured Matches (Priority Score >= 75)                                |  |
|  | [ Wales vs Portugal ]  [ Denmark vs Norway ]  [ Poland vs Sweden ]              |  |
|  +----------------------------------------------------------------------------------+  |
|  | SECTION 2: Live Matches (Pulsing Minute & Real-Time Score)                       |  |
|  +----------------------------------------------------------------------------------+  |
|  | SECTION 3: Grouped Competitions (Country Header + Logo + Accordion + Match Cards) |  |
|  |   v WORLD / EUROPE - UEFA Nations League (8 matches) [★]                         |  |
|  |     Card 1: Wales [crest] vs [crest] Portugal | 19:45 | [⚡ Prediction Available] |  |
|  |     Card 2: Denmark [crest] vs [crest] Norway  | 19:45 | [⚡ Prediction Available] |  |
|  |   v CONCACAF - CONCACAF Nations League (4 matches) [★]                           |  |
|  |   v AFRICA - Africa Cup of Nations Qualifiers (14 matches) [★]                    |  |
|  +----------------------------------------------------------------------------------+  |
|  | SECTION 4: Finished Matches (Full-Time Scores & Winner Markers)                  |  |
|  +----------------------------------------------------------------------------------+  |
|  | SECTION 5: Other Competitions (Collapsible Tier 4/5 Matches)                     |  |
+----------------------------------------------------------------------------------------+
```

---

### 3. Implementation Steps

1. **Prioritization Engine (`zola/services/prioritization.py`)**:
   - 0–100 match scoring formula based on configurable weights:
     - Competition importance: 30%
     - Competition popularity: 20%
     - Team popularity: 20%
     - Match relevance: 15%
     - Kickoff proximity: 10%
     - Data completeness: 5%
   - Built-in Tier definitions for domestic & international competitions.
   - Built-in popularity directory for clubs and national teams.
   - Categorization: Featured (80–100), High (60–79), Normal (35–59), Low (0–34).

2. **Repository Enhancements (`zola/database/repository.py`)**:
   - `get_discovery_fixtures()`: Queries fixtures with full team crests, competition logos, countries, scores, and prediction flags.
   - Integrates the prioritization engine to sort and group fixtures.
   - Computes category counts (All, Live, Upcoming, Finished, Postponed, Cancelled).
   - Manages user favorites (`user_favorites` table in SQLite).

3. **REST API Extensions (`zola/dashboard/server.py`)**:
   - `/api/discovery`: Returns full BeSoccer-style grouped payload with date navigation, counts, featured fixtures, live fixtures, and grouped competitions.
   - `/api/prioritization/config`: GET/POST for scoring weights.
   - `/api/favorites`: GET/POST for saving and retrieving team/league favorites.
   - `/api/sync`: Trigger on-demand sync for dates with rate-limit protection.

4. **Frontend UI Overhaul (`zola/dashboard/static/index.html`)**:
   - Horizontal date navigation bar with relative day tags, arrows, and date picker.
   - Category navigation tabs with dynamic count badges.
   - BeSoccer competition accordion cards with crests, logos, kickoff times in user timezone, and prediction pills.
   - Featured matches marquee grid.
   - Favorites system with local storage fallback.
   - Responsive dark/light theme CSS.

5. **Automated Testing & Verification**:
   - Unit tests for prioritization scoring and grouping logic.
   - API endpoint integration tests.
   - Verification across all 18 criteria.
