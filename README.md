# 🏎️ F1 Lap Time Predictor

A machine-learning project for predicting Formula 1 lap times using **tire age, grid position, and lap number**, with a leakage-free, stint-based evaluation strategy.

> **Selected race:** 2018 Monaco Grand Prix  
> **Best model:** Enhanced Random Forest  
> **Evaluation:** Held-out final stint per driver

## 🎯 Objective

The goal of this project is to predict in-race Formula 1 lap times from race and stint-level information.

The model uses three features:
- **Tire age** — number of laps completed on the current tire stint
- **Grid position** — driver's starting grid position
- **Lap number** — position within the race

Instead of randomly splitting individual laps, the project uses a **stint-based train/test split**. Earlier stint(s) for each driver are used for training, while the driver's final stint is held out for testing.

This provides a more realistic evaluation of how the model performs on unseen future laps within a race.

---

## 🔬 Methodology

### 1. Race selection

Candidate races through 2020 with lap-time and pit-stop data were screened using a dry/no-incident heuristic.

Because the supplied dataset does not contain direct weather, red-flag, or race-interruption information, this screening is treated as a heuristic rather than definitive proof of race conditions.

The **2018 Monaco Grand Prix (raceId=994)** was selected as the best qualifying candidate.

### 2. Data cleaning

For the selected eight drivers:

1. Remove laps containing pit stops.
2. Remove the lap immediately following each pit stop.
3. Split each driver's data by stint.
4. Use earlier stint(s) for training and the final stint for testing.
5. Calculate each driver's outlier threshold using **training data only**.
6. Apply the training-derived threshold to both training and test data.

**Cleaning result:**

`624 raw laps → 608 retained laps`

### 3. Feature engineering

| Feature | Description |
|---|---|
| `tire_age` | Tire age within the current stint |
| `grid` | Driver's starting grid position |
| `lap` | Race lap number |

The first retained lap after a pit-stop sequence is `P+2`, giving it `tire_age = 2`, because both the pit lap and immediate out-lap are removed.

### 4. Models

Two regression algorithms were evaluated:
- **Linear Regression**
- **Random Forest** — `n_estimators=200`, `max_depth=5`, `random_state=42`

Each model was evaluated using two feature configurations:

**Baseline**
- `grid`
- `lap`

**Enhanced**
- `grid`
- `lap`
- `tire_age`

---

## 📊 Results

| Model | Feature set | Train | Test | RMSE (ms) | MAE (ms) |
|---|---|---:|---:|---:|---:|
| Linear Regression | Baseline | 159 | 449 | 5335.9 | 3382.1 |
| Linear Regression | Enhanced | 159 | 449 | 5024.2 | 3020.6 |
| Random Forest | Baseline | 159 | 449 | 4226.0 | 2124.0 |
| **Random Forest ⭐** | **Enhanced** | **159** | **449** | **4207.5** | **2059.5** |

### Key findings

Adding `tire_age` improved both RMSE and MAE for both algorithms.

- Linear Regression RMSE: **5335.9 → 5024.2 ms**
- Random Forest RMSE: **4226.0 → 4207.5 ms**
- Best overall model: **Enhanced Random Forest**
- Best MAE: **2059.5 ms**
- Best RMSE: **4207.5 ms**

The Random Forest achieves lower error than Linear Regression, but its predictions flatten when the held-out final stint contains feature values outside the training range. This is expected because Random Forest models do not extrapolate beyond the feature ranges observed during training.

---

## 🔐 Leakage Prevention

Preventing data leakage is a central part of this project.

An earlier version calculated each driver's outlier threshold using both training and test laps. The corrected pipeline first creates the stint-based train/test split and then calculates the outlier threshold **from training rows only**.

That fixed threshold is subsequently applied to the test data.

The corrected pipeline was rerun from scratch, and the selected race, drivers, cleaning counts, feature construction, split sizes, and reported metrics remained numerically unchanged.

The race-ranking `cv` metric was also corrected to use:

```text
std(per-driver median lap times)
-------------------------------
mean(per-driver median lap times)
```

The corrected ranking continued to select the **2018 Monaco Grand Prix**.

---

## 📈 Prediction Visualization

The project includes a visualization comparing actual and predicted lap times for the selected test-set driver with the most held-out laps.

![Actual vs Predicted Lap Times](stint_prediction.png)

---

## 📁 Repository Structure

```text
f1-lap-time-predictor/
│
├── README.md
├── notebook.ipynb
├── requirements.txt
├── model_comparison.csv
├── run_metadata.json
├── stint_prediction.png
│
└── data/
    ├── results.csv
    ├── races.csv
    ├── lap_times.csv
    └── pit_stops.csv
```

---

## 🗃️ Dataset

The project uses Formula 1 World Championship data covering **1950–2020**.

The repository contains the four required CSV datasets:

- `results.csv`
- `races.csv`
- `lap_times.csv`
- `pit_stops.csv`

Pit-stop records begin in 2011, so the effective candidate race range for this project is **2011–2020**.

---

## ▶️ Reproducibility

Clone the repository and install the required dependencies:

```bash
pip install -r requirements.txt
```

The notebook expects the four datasets inside the `data/` directory.

Open `notebook.ipynb` and run the notebook from top to bottom.

A fixed random seed of **42** is used for reproducibility.

---

## ⚠️ Limitations

- Dry/no-red-flag status is inferred from lap-time behavior because the supplied data does not contain direct weather or incident information.
- Driver-level training sets are relatively small (**11–49 laps**).
- The reported metrics represent one executed run of the pipeline.
- Random Forest predictions cannot extrapolate beyond the feature range observed during training.
- The model focuses on a single selected race rather than attempting to generalize across all Formula 1 races.

---

## 🛠️ Tech Stack

- **Python**
- **Pandas**
- **NumPy**
- **Scikit-learn**
- **Matplotlib**
- **Jupyter Notebook**

---

## 📌 Project Focus

This project demonstrates:
- Time-series-aware model evaluation
- Feature engineering from race and stint data
- Regression modeling
- Model comparison
- Data cleaning and outlier handling
- Leakage prevention
- Reproducible machine-learning workflows

---

**Formula 1 Lap Time Predictor — built as a machine-learning project focused on realistic evaluation, feature engineering, and leakage prevention.**