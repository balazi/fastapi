# Model Card

For additional information see the Model Card paper: https://arxiv.org/pdf/1810.03993.pdf

## Model Details

Random forest classifier (scikit-learn `RandomForestClassifier`, 100 trees, `min_samples_leaf=5`, `random_state=42`) that predicts whether a person's annual income exceeds $50K. Categorical features are one-hot encoded; the label is binarized. The model, `OneHotEncoder` and `LabelBinarizer` are stored in `model/` as joblib files.

## Intended Use

Educational demonstration of an ML pipeline served through a FastAPI service. It must not be used for real decisions about employment, credit, housing or similar matters.

## Training Data

UCI Census Income dataset (`data/census.csv`, 32,561 rows, 14 features, label `salary`). Whitespace after the delimiters was removed. 80% of the rows (random split, seed 42) were used for training.

## Evaluation Data

The remaining 20% (6,513 rows) of the same dataset, encoded with the encoder and label binarizer fitted on the training split.

## Metrics

On the test split: precision 0.775, recall 0.617, F1 0.687.

Performance on slices of every categorical feature is in `model/slice_output.txt`. Examples:

| Slice | Count | Precision | Recall | F1 |
|---|---|---|---|---|
| sex = Female | 2126 | 0.771 | 0.464 | 0.579 |
| sex = Male | 4387 | 0.775 | 0.644 | 0.704 |
| race = White | 5595 | 0.778 | 0.617 | 0.688 |
| race = Black | 599 | 0.720 | 0.554 | 0.626 |
| race = Asian-Pac-Islander | 193 | 0.759 | 0.710 | 0.733 |

Slices with few samples give noisy estimates.

## Ethical Considerations

The data contains sensitive attributes (sex, race, native country) and reflects historical societal biases. Recall is notably lower for women and for some racial groups, so the model can systematically under-predict high income for them.

## Caveats and Recommendations

The data comes from a 1994 census and does not reflect current incomes. Slice metrics show uneven performance across groups, so audit and recalibrate before any real use. Only the hyperparameters above were used; no tuning or cross-validation was done.
