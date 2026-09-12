# F1 Lap Time Predictor

Predict in-race Formula 1 lap times from tire age, grid position, and lap number using a leakage-free, stint-based evaluation pipeline.

> Project: Formula 1 lap-time prediction  
> Selected race: 2018 Monaco Grand Prix  
> Best model: Enhanced Random Forest  
> Evaluation: Held-out final stint per driver

## Objective

The project predicts lap times within a single dry Formula 1 race using the Formula 1 World Championship (1950–2020) dataset. The modeling pipeline uses tire age, grid position, and lap number.

The evaluation deliberately avoids random row-level splitting. For each driver, earlier stint(s) are used for training and the final stint is held out for testing.

## Methodology

### 1. Race selection

Candidate races through 2020 with lap-time and pit-stop data were screened using a dry/no-incident heuristic. Because the supplied data has no weather, red-flag, or race-interruption field, this is explicitly treated as a heuristic rather than proof.

2018 Monaco Grand Prix (raceId 994) was selected as the best qualifying candidate.

### 2. Data cleaning

For the selected eight drivers:

1. Remove pit-stop laps.
2. Remove the lap immediately after each pit stop.
3. Split by stint: earlier stints → train, final stint → test.
4. Compute each driver's 1.5× median outlier threshold from training rows only.
5. Apply that train-derived threshold to both train and test.

Cleaning result: 624 raw laps → 608 retained laps.

### 3. Feature engineering

| Feature | Description |
|---|---|
| tire_age | Tire age within the current stint |
| grid | Driver's starting grid position |
| lap | Race lap number |

The first retained lap after a pit-stop sequence is P+2, giving it tire_age = 2, because both the pit lap and immediate out-lap are removed.

### 4. Models

Two regressors were evaluated using the same train/test rows:

- Linear Regression
- Random Forest — n_estimators=200, max_depth=5, fixed seed 42

Each was evaluated with a baseline feature set (grid + lap) and an enhanced feature set (grid + lap + tire_age).

## Results

| Model | Feature set | Train | Test | RMSE (ms) | MAE (ms) |
|---|---|---:|---:|---:|---:|
| Linear Regression | Baseline | 159 | 449 | 5335.9 | 3382.1 |
| Linear Regression | Enhanced | 159 | 449 | 5024.2 | 3020.6 |
| Random Forest | Baseline | 159 | 449 | 4226.0 | 2124.0 |
| Random Forest | Enhanced ⭐ | 159 | 449 | 4207.5 | 2059.5 |

### Key finding

Adding tire_age improved both metrics for both algorithms.

- Linear Regression RMSE: 5335.9 → 5024.2 ms
- Random Forest RMSE: 4226.0 → 4207.5 ms
- Best overall model: Enhanced Random Forest

The Random Forest has lower absolute error than Linear Regression, but its predictions flatten when the held-out final stint contains lap values beyond the training feature range. This is consistent with the model's inability to extrapolate beyond its training feature range.

## Leakage audit

A major part of this project is preventing preprocessing leakage.

An earlier version calculated each driver's outlier threshold using both train and test laps. The corrected pipeline first creates the stint-based split, calculates the threshold from training data only, and then applies that fixed threshold to the test data.

The corrected pipeline was rerun from scratch. The selected race, drivers, cleaning counts, feature construction, split sizes, and reported metrics remained numerically unchanged.

The race-ranking cv metric was also corrected to use the standard deviation of per-driver median lap times divided by the mean of that same population. The corrected ranking still selected the 2018 Monaco Grand Prix.

## Visualization

The repository includes stint_prediction.png, showing actual vs. predicted lap times for the selected test-set driver with the most held-out laps (driverId 1 / car #44).

![Actual vs predicted lap times](stint_prediction.png)

## Repository structure

    f1-lap-time-predictor/
    ├── README.md
    ├── notebook.ipynb
    ├── model_comparison.csv
    ├── run_metadata.json
    ├── stint_prediction.png
    └── requirements.txt

## Reproducibility

The notebook expects these four original F1 CSV datasets:

    results.csv
    races.csv
    lap_times.csv
    pit_stops.csv

Place them beside notebook.ipynb, or set the F1_DATA_DIR environment variable.

Then run the notebook from top to bottom.

A fixed random seed of 42 is used.

## Limitations

- Dry/no-red-flag status is inferred from lap-time behavior because the supplied data has no direct weather/incident field.
- Driver-level training sets are small: 11–49 laps.
- The results represent one executed run of the pipeline.
- Random Forest predictions cannot extrapolate beyond the training feature range.

## Interview talking points

If presenting this project, the strongest technical points are:

1. Why stint-based splitting? Random row splitting would allow information from the same stint/race progression to leak across train and test.
2. Why tire age? Lap number alone does not directly encode where a driver is within a tire stint; tire age adds that context.
3. How was leakage prevented? The outlier threshold is fitted using training rows only.
4. Why Random Forest? It achieved the lowest RMSE and MAE among the evaluated models.
5. What is the biggest limitation? The model is trained on relatively small per-driver samples and Random Forest cannot extrapolate outside the observed feature range.

## Dataset

Formula 1 World Championship data covering 1950–2020, using results.csv, races.csv, lap_times.csv, and pit_stops.csv. Pit-stop records begin in 2011, so the effective candidate range for this task is 2011–2020.

---

Built as a machine-learning project focused on time-series-aware evaluation, feature engineering, and leakage prevention.
