# NIAI Earthquake + scikit-learn Lab (Day 1–3)

This is a beginner-friendly, combined project folder. The notebook and earthquake application are **separate pieces of work** in the same ZIP.

## Folder map

- `notebooks/day2-3_part1_sklearn_basics.ipynb` — the completed learning notebook. It uses scikit-learn's built-in **diabetes** dataset for regression, the built-in breast-cancer dataset for classification, and iris for multiclass classification. You do not need to download a separate diabetes CSV.
- `quakes/` — the earthquake application: fetch data, clean it, create features, save it to SQLite, train models, and evaluate them.
- `tests/` — automated checks for both Day 1 and Day 2–3 functions.
- `data/stream/` — optional JSONL stream files.
- `data/fallback/` — place the instructor's `all_week.geojson` here if you need offline fallback. The starter ZIP did not include that data file.
- `data/processed/` — generated CSVs, preprocessor, and SQLite database.
- `models/` and `reports/` — saved trained pipelines and JSON evaluation reports.
- `ANSWERS.md` and `ANSWERS_ML.md` — answers to the questions present in the supplied files.
- `README_PART2.md` — ETL, training, and scheduling instructions.

## 1. Install dependencies

From the project root, activate your Python environment and run:

```powershell
python -m pip install -r REQUIREMENTS.txt
```

The supplied `environment.yml` also lists a Conda environment named `niai-ml`. If Conda is not installed, use the pip command above.

## 2. Run the notebook separately

Open the project in VS Code or JupyterLab, open `notebooks/day2-3_part1_sklearn_basics.ipynb`, and run all cells from top to bottom. The notebook was executed successfully and its assertions passed when packaged.

## 3. Run project tests

```powershell
pytest -q
```

The supplied test suite currently passes **23 tests**.

## 4. Run the earthquake ETL

```powershell
python -m quakes.etl --mode batch
```

This downloads the USGS weekly feed (or uses `data/fallback/all_week.geojson` if the download fails and that file exists), cleans the data, creates features and the `big_quake` target, and writes the results to `data/processed/quakes.db`.

To merge in saved streaming files too:

```powershell
python -m quakes.etl --mode merge
```

## 5. Train and evaluate a model

Regression predicts earthquake magnitude (`mag`):

```powershell
python -m quakes.train --task regression --model linear
python -m quakes.train --task regression --model poly --degree 3 --alpha 10
```

Classification predicts whether `big_quake` is 1 (`mag >= 4.5`):

```powershell
python -m quakes.train --task classification --model logistic
python -m quakes.train --task classification --model logistic --balanced
```

Training compares the selected model with a dummy baseline, cross-validates on training data only, and saves the fitted pipeline under `models/` and a JSON report under `reports/`. For classification, the threshold is selected from out-of-fold training predictions rather than from the test set.

## Important project concepts

1. **ETL** means extract data, transform/clean it, then load it into SQLite.
2. **Features (`X`)** are the information the model uses; **target (`y`)** is what it predicts.
3. `mag` is removed from the classification features, and both `mag` and `big_quake` are removed from the model inputs to prevent target leakage.
4. Imputers, scalers, and one-hot encoders live inside an sklearn pipeline. They are fitted on training data during model fitting, not on the test set.
5. A dummy model is a baseline. A more complex model should be compared with it rather than judged alone.
6. A model score is an estimate, not a guarantee that a future earthquake can be predicted accurately.

## Lab handout note

The supplied notebook has three checkpoint questions, which are answered. `README_PART2.md` refers to a separate handout containing six questions, but that handout was not included in the provided ZIP. `ANSWERS_ML.md` therefore answers the three questions actually supplied rather than guessing the missing questions.
