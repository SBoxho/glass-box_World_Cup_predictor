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

### FIFA men's world ranking
- **History source:** [`Dato-Futbol/fifa-ranking`](https://github.com/Dato-Futbol/fifa-ranking) —
  a community compilation of the public FIFA/Coca-Cola Men's World Ranking
  (`team, total_points, date`), Dec 1992 → Sept 2024.
- **Access:** downloaded at build time from the public GitHub raw mirror by
  [`core/ranking.py`](core/ranking.py); cached under `data/raw/` (gitignored). **No API key required.**
  Not committed as a raw dump.
- **Current snapshot (committed):** [`data/fifa_ranking_2026.json`](data/fifa_ranking_2026.json) —
  the official **11 June 2026** ranking points for the WC-2026 teams in the top 20 (small, factual,
  public data). Appended to the history as one more dated snapshot so 2026 predictions use current
  points; any WC team not in the snapshot falls back to its latest historical ranking (~Sept 2024).
- **License:** the underlying ranking is published by FIFA (public facts); the compilation is a
  community dataset, used here for educational purposes with attribution.
- **Use here:** attached **point-in-time** (the latest ranking dated on/before each match — leak-free,
  see `core.ranking.points_as_of`) as the `fifa_points_diff` model feature and a third baseline
  (FIFA-only) in [`core/model.py`](core/model.py).
- **Caveats:** (1) FIFA changed its points method in 2018 (SUM → Elo-based), so absolute points are
  not comparable across that boundary — the model only ever uses a *same-date* difference between two
  teams, so within-match scale is consistent, but the feature's distribution shifts across eras.
  (2) The free history feed ends Sept 2024; recent matches before the 11 Jun 2026 snapshot use the
  latest pre-2024 ranking. (3) FIFA points are **strongly correlated with Elo** (Pearson r ≈ 0.74 on
  the training set) — the feature adds little independent signal; this is reported honestly in the
  app's "Under the Hood" tab (the FIFA-only baseline is in fact slightly weaker than Elo-only).

### EA Sports FC / FIFA squad strength (Phase 3)
A per-roster strength signal that Elo structurally misses, built from **versioned** in-game player
ratings and attached **point-in-time** (a match dated `D` uses only the ratings *version* released
on/before `D` — leak-free, see `core.squads.strength_as_of`). Four model features (home − away):
`squad_strength_diff` (best-XI mean overall), `attack_vs_def` (attack-line vs defence-line matchup),
`depth_diff` (players 12–26), `star_power_diff` (top-3). Two tiers, mirroring the FIFA ranking:

- **History source (versioned, FIFA 15 → FIFA 23):**
  [`jsulz/FIFA23`](https://huggingface.co/datasets/jsulz/FIFA23) on Hugging Face — a public mirror of
  the community **sofifa-derived "legacy" complete-player dataset** (`male_players (legacy).csv`:
  one row per player per FIFA version, with `fifa_version`, `fifa_update_date`, `nationality_name`,
  `overall`, `player_positions`, `pace/shooting/passing/dribbling/defending/physic`). Originally
  compiled by [stefanoleone992](https://www.kaggle.com/datasets/stefanoleone992/ea-sports-fc-24-complete-player-dataset)
  from the publicly scrapable sofifa.com.
  - **Access:** downloaded at build time by [`core/squads.py`](core/squads.py); cached under
    `data/external/` (gitignored). **No API key required. Not committed as a raw dump.**
- **Current snapshot (committed):** [`data/squads2026.json`](data/squads2026.json) — the 48 WC-2026
  squads (top-26 by overall per nation, with the per-attribute fields the UI shows), derived from
  the public **EA FC 26** ratings
  ([`ismailoksuz/EAFC26-DataHub`](https://github.com/ismailoksuz/EAFC26-DataHub), same sofifa-derived
  legacy schema, `fifa_version == 26`) by
  [`scripts/build_squads_snapshot.py`](scripts/build_squads_snapshot.py). Appended as the latest
  version so 2026 predictions use current squads. Small, factual, committed.
- **License:** the underlying ratings are EA SPORTS FC in-game data; the datasets are community
  compilations of the publicly scrapable sofifa.com, used here for **educational / non-commercial**
  purposes with attribution. Only small derived artifacts are committed.
- **⚠️ In-game ratings are THIRD-PARTY ESTIMATES, not official data** — stated in the UI (near the
  squad panel and the ablation) and here.
- **Use here:** the four squad features + a **squad-only** baseline and a **with-vs-without-squad
  ablation** in [`core/model.py`](core/model.py); the squad panel + radar in the Match Predictor.
- **Caveats / coverage gaps:** (1) The free history runs FIFA 15 (Sep 2014) → FIFA 23 (Sep 2022),
  one snapshot per version at its launch roster; matches before Sep 2014 and the 2023→2026 gap fall
  back to the nearest available version (FIFA 23, then the FC 26 snapshot). On the training set the
  squad feature is non-zero on ~31% of rows (the modern, rated portion). (2) EA under-represents
  smaller nations — a few WC-2026 sides have <26 (down to 3) rated players, so their depth/line
  metrics are noisier; nations absent from a version fall back to a fixed `SQUAD_OVR_BASE` (0 diff).
  (3) Squad strength is **moderately collinear with Elo** (Pearson r ≈ 0.53 on the training set) —
  the with-vs-without ablation shows only a **tiny lift** (≈ +0.0014 log-loss, +0.4pp accuracy) and
  the model still essentially ties the Elo-only baseline. Reported honestly in "Under the Hood".

---

## Planned enhancement (not yet integrated)

- **Squad market value** (Transfermarkt-derived datasets) — a strong real-world strength proxy with
  a historical time series, complementary to in-game ratings. Use existing datasets only; **do not
  scrape live** (ToS). When integrated it will be loaded via a documented download step and **never
  committed** as a raw dump, with its specific licensing note added here.
- **Player-level explainability** — surfacing which players drive a nation's squad-strength feature.

---

## Disclaimer

Outputs are **probabilistic** and for educational / decision-support purposes only — **not betting
advice**. In-game ratings, where used, are third-party estimates and not official measurements.
