# Data sources & attribution

This project is **educational / non-commercial**. All data is loaded via documented, scriptable
download steps — **no raw third-party dumps are committed** to the repository.

## Used by the core app

### International match results
- **Source:** [`martj42/international_results`](https://github.com/martj42/international_results)
  — a community-maintained dataset of every men's international football match since 1872
  (`date, home_team, away_team, home_score, away_score, tournament, city, country, neutral`).
- **Access:** downloaded at build time from the public GitHub raw mirror by
  [`core/ingest.py`](core/ingest.py); cached under `data/raw/` (gitignored). No API key required.
- **License:** released into the public domain (CC0) by the maintainers. Attribution given here as
  a courtesy.
- **Use here:** drives the rolling Elo ratings, all point-in-time features, and model training.

### 2026 tournament structure
- **Source:** the official FIFA World Cup 2026 final draw (held 5 December 2025, Washington DC),
  cross-checked across multiple outlets (FIFA, ESPN, Al Jazeera, NBC Sports).
- **Stored in:** [`data/wc2026.json`](data/wc2026.json) — the 12 groups, host assignments
  (USA / Mexico / Canada), and the Round-of-32 bracket template (committed; small, factual).

### Live 2026 results
- **Source:** [`openfootball/worldcup.json`](https://github.com/openfootball/worldcup.json) —
  community-maintained match data for the 2026 World Cup
  (`2026/worldcup.json`: `team1, team2, score.ft, group, round, date`).
- **Access:** fetched on demand from the public GitHub raw mirror by
  [`core/live.py`](core/live.py); cached under `data/raw/` (gitignored). **No API key required.**
  The fetch is optional — the app works fully offline, simulating from the pre-tournament state if
  no results are loaded.
- **License:** released into the **public domain (CC0)** by the maintainers.
- **Use here:** the played *group-stage* results are run through `config.normalize_team` and locked
  into the simulator's `known_results` block so the Tournament Simulator runs forward from the
  current standings (knockout matches are re-simulated, not locked). Populate the committed file
  with `python scripts/update_results.py --write`, or refresh on demand via the app.
- **Caveat:** scores reflect upstream community freshness and may lag or contain placeholder data —
  verify against official results before trusting a committed snapshot. No raw dump is committed.

---

## Planned enhancement (not yet integrated)

The roadmap includes a **current-squad strength** signal — a per-roster feature that Elo
structurally misses — plus a player-level explainability layer. These rely on proprietary /
ToS-restricted datasets, so when integrated they will be loaded via a documented download step and
**never committed** as raw dumps:

- **EA FC 26 player ratings** (community datasets) — global 0–99 overall ratings, comparable across
  leagues. In-game ratings are **third-party estimates, not official data**.
- **Squad market value** (Transfermarkt-derived datasets) — a strong real-world strength proxy
  with a historical time series. Use existing datasets only; **do not scrape live** (ToS).
- **Historical versioned ratings** (FIFA 15 → FC 26) — required to build a *time-consistent*
  per-year nation strength index (no leakage: a match uses only the rating version dated on/before it).

When this module lands, each source will be cited and linked here with its specific licensing note,
and an ablation table (model with vs without squad features) will be reported transparently in the
app's "Under the Hood" tab.

---

## Disclaimer

Outputs are **probabilistic** and for educational / decision-support purposes only — **not betting
advice**. In-game ratings, where used, are third-party estimates and not official measurements.
