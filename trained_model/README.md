# NeuroX - Ticket Classification Model

NeuroX is a customer-support ticket classifier for TensorForge 2.0. This folder contains the model-development work for Task 1: preparing the data, training and comparing models, evaluating predictions, and saving the models for the backend teammate.

The current model version is **`neurox-v1`**. It predicts a main category, an optional secondary category, urgency, and the support team that should receive the ticket.

## Files

```text
NeuroX/
  README.md
  TensorForge2.0_Phase2_MVP_Guidelines.pdf
  dataSets/
    DATA_NOTES.md
    train.csv
    train.jsonl
    validation.csv
    validation.jsonl
  appi nad schema/
    ...provided API specification and JSON schemas
  trained_model/
    model_dev.ipynb
    saved_models/
      neurox-v1/
        predict.py
        models.joblib
        requirements.txt
        validation_metrics.json
```

- `model_dev.ipynb`: data preparation, training experiments, evaluation, and saving.
- `models.joblib`: all three trained pipelines, including their TF-IDF vectorizers, model version, team mapping, and package versions.
- `predict.py`: loads the saved models and provides `predict_ticket()` for inference.
- `requirements.txt`: package versions used by the saved models.
- `validation_metrics.json`: scores for the final combined prediction function.

Keep the four files in `neurox-v1` together. The notebook saves the model there, and `predict.py` loads `models.joblib` from beside its own file.

## Dataset and cleaning

The supplied synthetic RideEat dataset contains **4,000 training tickets** and **800 validation tickets**. It includes English, Sinhala, Tamil, Singlish, Tanglish, and mixed-language tickets. The CSV and JSONL files contain the same records, so the notebook reads the CSV files only.

The notebook:

1. Reads the CSV files using UTF-8 to preserve multilingual text.
2. Replaces missing subjects with empty strings.
3. Replaces missing secondary categories with empty strings, then uses `no_secondary` as the training label for tickets without a second issue.
4. Checks for duplicate ticket IDs.
5. Keeps training and validation data separate.

The dataset review also found no missing ticket bodies, duplicate IDs, or shared IDs between the two splits. Cleaning happens in memory; the original dataset files remain unchanged. Separate cleaned CSV files are not required to run the notebook.

## How the models work

Each ticket is prepared in this exact format during both training and prediction:

```text
channel [SUBJECT] subject [TICKET] text
```

Only `channel`, `subject`, and `text` are model inputs. Ticket IDs are echoed in responses but are not used for training. Language labels are used for analysis, not as prediction inputs.

All three models use character TF-IDF features and Logistic Regression:

- Character patterns of 2 to 5 characters: `ngram_range=(2, 5)`.
- Patterns must appear in at least two training documents: `min_df=2`.
- Class weighting: `class_weight="balanced"`.
- Maximum training iterations: `max_iter=2000`.

| Model | Training target | Final `C` setting |
|---|---|---:|
| Main category | `category` | 100.0 |
| Urgency | `is_urgent` | 1.0 |
| Secondary category | `secondary_category`, including `no_secondary` | 1.0 |

The main-category model was selected using three-fold cross-validation on the training set. The search compared `C` values of 0.1, 1.0, 10.0, and 100.0, choosing the highest average macro F1. Each fold fitted its own text vectorizer. The selected model was then trained on all 4,000 training tickets.

## Experiments and results

| Main-category experiment | Validation accuracy |
|---|---:|
| Original character model, `C=1.0` | 53.25% |
| Combined word and character features, `C=1.0` | 52.75% |
| Character model with stronger regularization, `C=0.1` | 41.50% |
| Selected character model, `C=100.0` | **58.63%** |

The saved model uses the selected character model. Its validation accuracy improved by about **5.38 percentage points** over the original model.

The following results come from the final combined function, including its response rules, on all 800 validation tickets:

| Measurement | Result |
|---|---:|
| Main-category accuracy | **58.63%** (469/800) |
| Main-category macro F1 | **0.609** |
| Urgency accuracy | **96.12%** (769/800) |
| Secondary-category accuracy overall | **95.50%** (764/800) |
| Secondary accuracy on tickets with a second issue | **55.00%** (44/80) |
| Responses passing the supplied response schema | **800/800** |

The saved models were reloaded successfully. The standalone prediction function was also checked against the notebook function on all 800 validation tickets, with matching results.

The notebook's earlier per-language table, misclassified-ticket list, and confusion matrix describe the **original baseline model**. Use the final combined evaluation and `validation_metrics.json` for the saved model's overall scores.

## Run the saved model

The model was created and tested with **Python 3.14.6**. Its package versions are recorded in `requirements.txt`. Use these versions when loading the saved model.

From the `NeuroX` project folder, install the packages and run the example:

```powershell
python -m pip install -r trained_model/saved_models/neurox-v1/requirements.txt
python trained_model/saved_models/neurox-v1/predict.py
```

This loads the existing models and prints a prediction for the example ticket in the script. It does not retrain the models. Prediction needs no external API or model download once the packages and saved model are available.

If PowerShell reports that Python was not found on the development computer, use the installed interpreter directly:

```powershell
$neuroxPython = "C:\Users\Muqshith Akbar\AppData\Local\Python\pythoncore-3.14-64\python.exe"
& $neuroxPython -m pip install -r trained_model/saved_models/neurox-v1/requirements.txt
& $neuroxPython trained_model/saved_models/neurox-v1/predict.py
```

On another computer, use that computer's Python installation instead of this local path.

## Use it in the backend

Keep `predict.py` and `models.joblib` beside each other in a folder your backend can import from. Install the recorded requirements in the backend's Python environment.

```python
from predict import predict_ticket

result = predict_ticket(
    channel="chat",
    text="I was charged twice for the same ride.",
    subject="",
    ticket_id="demo-001",
)
```

The models load once when the module is imported. The example at the bottom of `predict.py` runs only when the script is executed directly.

The returned dictionary contains `category`, `secondary_category`, `team`, `is_urgent`, `confidence`, and `model_version`. It also includes `ticket_id` when supplied.

The function applies these rules:

- A missing secondary category becomes `None`, which becomes `null` in JSON.
- A secondary category matching the primary category becomes `None`.
- Spam is always nonurgent and has no secondary category.
- The team comes from the competition's fixed category-to-team mapping.

Accepted channels are `email`, `chat`, and `call_transcript`. Ticket text must contain a non-whitespace character and be at most 10,000 characters. Subject length is limited to 500 characters, and a supplied ticket ID is limited to 64 characters. The function raises `ValueError` for these invalid inputs.

The backend teammate must implement the HTTP endpoints, full request-schema validation, error responses, authentication, batch jobs, and deployment. The model function supplies predictions for those endpoints.

## Run the training notebook again

1. Install the model requirements above.
2. Install the notebook kernel and schema-checking dependency in the same Python environment:

   ```powershell
   python -m pip install ipykernel jsonschema==4.26.0
   ```

   If using the explicit interpreter variable above, replace `python` with `& $neuroxPython`.

3. Open `trained_model/model_dev.ipynb` in VS Code with Jupyter support and select that Python environment as the kernel.
4. Ensure the notebook's working directory is `trained_model`, because it reads the data from `../dataSets`.
5. Restart the kernel and run the cells from top to bottom.

Running all cells retrains the models, evaluates them, and replaces the saved model, requirements, and metrics in `saved_models/neurox-v1`.

## Limitations

- Main-category accuracy is 58.63%, so 331 of the 800 validation tickets were assigned the wrong main category. This is a working first model with room for improvement.
- 720 of 800 tickets have no secondary category. The 95.50% overall secondary score should therefore be reported alongside the **55.00%** score on tickets that actually have a second issue.
- The urgency model identifies 65 of the 80 urgent tickets and misses 15; high overall accuracy does not mean every urgent ticket is detected.
- Training cross-validation macro F1 was about 0.976, but final validation macro F1 is 0.609. Report these separately. Validation results were inspected during development, so they are not an untouched final-test estimate.
- `confidence` is the main classifier's probability estimate. It has not been separately calibrated and is not a guarantee that a prediction is correct.
- The dataset is synthetic. These results do not establish performance on real customer tickets.
