# Review Classifier

A Python project that trains a TF-IDF + Logistic Regression text classifier for movie/TV reviews, tunes hyperparameters with randomized cross-validation, evaluates on a holdout split, and generates test-set predictions.

## Features

- Reads training and test data from CSV files.
- Validates required columns before training.
- Converts review text into TF-IDF word and bigram features.
- Tunes model hyperparameters with `RandomizedSearchCV`.
- Evaluates performance using macro F1 on an untouched validation split.
- Retrains the best model on all labeled data before test prediction.
- Saves predictions, metrics, best parameters, a text report, and the trained model.

## Requirements

- Python
- pip

Install dependencies:

```bash
pip install scikit-learn pandas numpy joblib
```

## Expected files

### `train.csv`
Must contain these columns:

- `ID` - unique row identifier
- `TEXT` - review text
- `LABEL` - binary class label, typically `0` or `1`

### `test.csv`
Must contain these columns:

- `ID` - unique row identifier
- `TEXT` - review text

## Project structure

```text
.
├── train_reviews.py
├── train.csv
├── test.csv
└── output/
```

## How to run

Run:

```bash
python train_reviews.py --train train.csv --test test.csv --outdir output
```

Example with custom settings:

```bash
python train_reviews.py --train train.csv --test test.csv --outdir results --cv 5 --n-iter 15
```

## Command-line arguments

| Argument | Default | Description |
|---|---|---|
| `--train` | `train.csv` | Path to the training CSV |
| `--test` | `test.csv` | Path to the test CSV |
| `--outdir` | `output` | Directory where artifacts are written |
| `--cv` | `5` | Number of cross-validation folds |
| `--n-iter` | `15` | Number of randomized search iterations |

## Output files

After a successful run, the output directory will contain:

- `submission.csv` - predicted labels for the test set
- `model.joblib` - trained scikit-learn pipeline
- `metrics.csv` - best cross-validation F1 and heldout F1
- `best_params.txt` - best hyperparameters found during search
- `report.txt` - classification report and confusion matrix

## Workflow summary

1. Load and validate `train.csv` and `test.csv`.
2. Fill missing text values with empty strings.
3. Split labeled data into training and validation sets.
4. Build a pipeline with text selection, TF-IDF, and Logistic Regression.
5. Tune hyperparameters using randomized search with stratified cross-validation.
6. Evaluate the best model on the untouched validation split.
7. Retrain the best model on the full labeled dataset.
8. Predict labels for the test set and save all artifacts.

## Notes

- The project assumes a binary classification task.
- Input CSV paths are relative to the working directory.
- The trained model is saved with `joblib`, so it can be loaded later.

## Example output

Typical console output looks like:

```text
Best CV F1 macro (train only): 0.92452
Holdout F1 macro (true generalization): 0.92227
Wrote: output\submission.csv
Wrote: output\model.joblib
Wrote: output\metrics.csv
Wrote: output\best_params.txt
Wrote: output\report.txt
```
