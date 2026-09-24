# Fantasy Football Capstone

[GitHub repository](https://github.com/mboem66/AI-powered-fantasy-NFL-system)

A Python project for predicting weekly fantasy football performance and explaining
start/sit decisions through a future web app. The submitted task breakdown in
`docs/proposal/` is the project scope.

## Initial decisions

- Historical seasons: **2023-2025**; update season: **2026**.
- Prediction positions: **QB, RB, WR, TE**.
- Scoring: **full PPR**. Scoring constants are recorded in `config/project.toml`.
- Source: **nflverse**, accessed through its Python package **nflreadpy**.
- Storage: compressed **Parquet** for source tables and later model datasets.

## Folder structure

```text
Capstone Project/
|-- config/project.toml       Seasons, positions, and scoring settings
|-- config/columns.toml       Columns retained for each source dataset
|-- src/fantasy_football/
|   |-- config.py             Load settings and resolve project paths
|   |-- data/                 Task 1: download and prepare data
|   |-- features/             Task 1: future pregame feature transformations
|   |-- modeling/             Tasks 2, 3, 6: training and evaluation
|   |-- recommendations/      Task 4: start/sit and explanations
|   `-- rag/                  Task 5: retrieval and grounded answers
|-- app/                      Future web application
|-- data/
|   |-- raw/                  Source tables, organized by dataset and season
|   |-- processed/            Future cleaned tables and player-week features
|   `-- manifests/            Download reports with timestamps and schemas
|-- artifacts/                Future saved models and evaluation results
|-- notebooks/                Exploration; reusable logic belongs in src/
|-- tests/                    Automated correctness checks
|-- docs/                     Task checklist, sources, and column guide
|-- .vscode/                  Local interpreter and test settings
`-- pyproject.toml            Python dependencies and package configuration
```

The project structure separates data processing, feature engineering, modeling,
recommendations, and retrieval into dedicated folders.

## Task 1: NFL Data Collection and Feature Engineering

Completed work includes the data download pipeline, Parquet storage, column
selection, and the historical player-game table described below.

### Data collection

The initial pipeline collected **29 Parquet files**, covering player statistics,
team statistics, schedules, rosters, snap counts, and player/team reference data.
The active historical range is **2023-2025**. Schedules and rosters for **2026** were also
available at the initial download. The pipeline supports current-season refreshes
as new results become available and records download outcomes for review.

### Column selection

Each dataset was reduced to fields relevant to the project. Redundant fields,
such as weekday when the game date is already present, were removed. Player
statistics retain a broader set of offensive production and advanced metrics to
support later feature engineering.

| Dataset | Original columns | Retained columns |
| --- | ---: | ---: |
| Schedules | 46 | **21** |
| Player statistics | 150 | **65** |
| Team statistics | 138 | **38** |
| Rosters | 36 | **11** |
| Player reference | 39 | **7** |
| Team reference | 16 | **5** |
| Snap counts | 16 | **11** |

Column selection preserved all row counts across the 29 files and reduced local
storage from approximately **9.5 MB to 4.4 MB**. The retained fields are defined in
[the column configuration](config/columns.toml)

### Historical player-game dataset

Combined regular-season **QB, RB, WR, and TE** statistics with game information
and available snap counts into one Parquet dataset. Each row represents one
player in one game. Full-PPR points were calculated and checked against nflverse.

### Recent performance and upcoming games

Added previous-game PPR points and three-game averages for PPR points, targets,
carries, and offensive snap percentage. Opponent context includes recent points
allowed to the player's position. Features use earlier games only.
Usage trends show changes in targets, carries, and snap share from the most recent
appearance compared with the preceding two appearances.

Upcoming-game records include active players with recent offensive participation.
One update command refreshes
the data and rebuilds both historical and upcoming-game datasets.
