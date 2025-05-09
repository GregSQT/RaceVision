# RaceVision

RaceVision is a deep learning pipeline for lap-by-lap prediction of Formula 1 races. It uses historical race data from the Ergast Motor Racing Database to build time-series datasets and trains an LSTM model to forecast driver positions, lap times, and pit-stop events for all 20 grid starters.

---

## 📁 Repository Structure

```
RaceVision/
├── scripts/
│   ├── 1-DataPreparation.py          # Converts raw Ergast DB CSVs to per-race/lap CSVs
│   ├── 2-RacePredictionPipeline.py   # Builds embeddings, .npy datasets, and defines & trains LSTM model
├── db/                               # Raw and processed data directory (user‐specified)
│   ├── races.csv                     # Metadata: race calendar
│   ├── lap_times.csv                 # Raw lap‐time logs
│   ├── results.csv                   # Race results per driver
│   ├── qualifying.csv                # Qualifying session times
│   ├── pit_stops.csv                 # Pit‐stop events (from 2012)
│   ├── driver_standings.csv          # Championship standings per round
│   ├── constructor_standings.csv     # Constructor standings per round
│   ├── races/                        # Generated per‐race CSVs by script 1
│   │   └── <year>/race<raceId>.csv
│   ├── drivers_short.csv             # Unique driverId listing
│   └── races_npy/                    # Generated .npy inputs/outputs by script 2
│       └── <year>/{idx}_in.npy
│                     {idx}_exp.npy
├── sd/                               # Model checkpoints and final weights
├── README.md                         # (this file)
```

---

## 🚀 Quickstart

1. **Clone the repository**

   ```bash
   git clone https://github.com/yourusername/RaceVision.git
   cd RaceVision
   ```

---

## **1-DataPreparation**

### ** Script preparation**

1. **Modules installation**

   * Uncomment the "pip install" commands in the initialization block.

2. **Configure data directory**

   * Edit `db_dir` path at the top of each script (usually `E:\Dropbox\Informatique\Holberton\F1_Project\db`).

3. **Run 1-DataPreparation**

   * Filter races from 2018 onwards (excluding 2021)
   * Build `db/races/<year>/race<raceId>.csv` with one row per lap, columns for each driver’s grid position, lap time, pit flag, status, and championship standings.
   * Generate `db/drivers_short.csv` listing all unique driverId values.
   * Post-process pit flags to carry-over before-stop lap.


---

## **2-RacePrediction**

1. **Generate embeddings & `.npy` datasets**

   This script:

   * Defines helper functions to embed status codes, lap times, driver IDs, and circuits into fixed-length vectors.
   * Iterates per‐race CSVs, builds input (`*_in.npy`) and expected-output (`*_exp.npy`) arrays for each lap.
   * Saves them under `db/races_npy/<year>/`.

2. **Train the LSTM model**

   * run_all, model, ds, crit, opt, sched, device; run_all(model, ds, crit, opt, sched, device, epochs=10)"
   * Checkpoints saved every 5 epochs in `sd/`.
   * Final model weight stored at end of training.

3. **Evaluate & predict**

   * run_test, model, ds, crit, device; print('Test Loss:', run_test(model, ds, crit, device))"
   * ds, model, pos_df; ds.set_year(2020); ds.set_round(10); p, _ = ds[10]; out, _ = model(p.unsqueeze(0).unsqueeze(0).float().to(device), model.zero_states()); print(pos_df(p,  out.squeeze()))"

---

## 🧩 Key Components

### 1. Data Preparation (script `1-DataPreparation.py`)

* **Raw CSVs**: Loaded from Ergast DB (`races.csv`, `lap_times.csv`, `results.csv`, etc.)
* **Race selection**: Filters seasons ≥2018, reserves 2021 for testing.
* **Per-race CSV**: One row per lap, features for each of 20 drivers:
  * `driverId`, `driverStanding`, `constructorStanding`, `position`, `inPit`, `status`, `laptime`
  * Lap 0 captures qualifying (Q1/Q2/Q3 times converted to seconds).
* **Drivers list**: `drivers_short.csv` of unique driverIds.
* **Pit flag fix**: Ensures `inPit` is active on the lap preceding the stop.

### 2. Embedding & Dataset Generation (script `2-RacePredictionPipeline.py`)

* **Helper functions**:
  * `time_to_int`: Convert "MM\:SS.ms" → seconds.
  * `status_embed` / `stat_unbed`: One-hot encoding / uncoding of status categories.
  * `lapTime_embed` / `lt_unbed`: Encode / uncode lap time into 32‑dim features.
  * `driver_embed_idx` / `driver_embed` / `driver_unbed`: Map driverId ↔ one-hot index.

* **.npy creation**:
  * **Input features (`*_input.npy`)**: 4051‑dim vector per lap:
    * 130-dim circuit one-hot, lap\_pct, then per-driver blocks:
      * Driver one-hot (130), standings flags, position one-hot (21), pit flag, status (6), laptime (32), random placeholder.
  * **Expected outputs (`*_expected.npy`)**: For next lap:
    * Position one-hot (21), normalized laps-to-pit, status (6), laptime (32) → per driver.

* **RaceDataset**: Custom `Dataset` class loading one race at a time, caching, with `__len__` and `__getitem__` yielding `(input, target)` tensors.

### 3. Model & Training

* **`RacePredictionModel`**:

  * Stacked LSTM (`lstm_layers`, `lstm_hids`) → hidden → fully-connected → `output_size`.
  * Xavier initialization for weights, zero biases.

* **Training utilities**:

  * `run_train`: Truncated BPTT per-lap, MSE loss, Adam optimizer, LR scheduler.
  * `run_test`: Evaluate on reserved season (≥2020).
  * `run_all`: Loop epochs, call train/test, save checkpoints.

* **Inference & Demo**:

  * Load latest checkpoint, set dataset to a specific year/round, call `model(p)` → decode via `pos_df` to display a DataFrame of predicted positions, pit stops, statuses, and lap times.
  * Batch inference & rolling prediction for *n* future laps.

---

## ⚙️ Configuration

* **`db_dir`**: Path to data folder (must contain raw Ergast CSVs).
* **GPU**: Automatically detects CUDA; set `device = torch.device("cuda" if available else "cpu")`.
* **Hyperparameters**: Modify LSTM hidden size, layers, dropout, learning rate, epochs in the training section of `2-RacePredictionPipeline.py`.

---

## ✅ TODO / Future Work

* Integrate weather, tyre compounds, track surface data.
* Model fuel‐weight impact when regulations change.
* Explore Transformer-based architectures for improved long‑term modeling.

---