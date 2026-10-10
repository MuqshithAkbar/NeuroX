# NeuroX Ticket Classifier

NeuroX is a prototype ticket classifier and routing system for TensorForge 2.0 that classifies the ticket into a category, optionally a secondary category, urgency level, and the support team assigned to it..

**Model version:** `neurox-v1`

## What it includes

- A trained text-classification system.
- FastAPI backend with endpoints for single ticket, batch, and asynchronous batch job
- A simple web-based user interface
- The provided API schemas and OpenAPI definition

## Project layout

```text
NeuroX/
├── API specs and schemas/       # API contract and JSON schemas
├── backend/                     # FastAPI service
├── dataSets/                    # Training and validation data
├── frontend/                    # Browser demo
├── trained_model/
│   ├── model_dev.ipynb          # Training and evaluation notebook
│   └── saved_models/neurox-v1/  # Saved model, predictor, metrics
├── Dockerfile
└── README.md
```

## Run the API with Docker

First, install and launch Docker Desktop. Run the following commands in the NeuroX project root folder.

Build the image:

```powershell
docker build -t neurox-api .
```

Initiate the API by substituting the blank with the approved team API key:
```powershell
docker run --rm -p 8000:8000 -e "API_KEY=YOUR_TEAM_API_KEY" neurox-api
```

Leave this terminal window open as you use the API. Do not place the actual key in this README file, source code, or Git repository..

Ensure the readiness of the service:
- Health check: `http://127.0.0.1:8000/health`
- Interactive API page: `http://127.0.0.1:8000/docs`

Health endpoint is public. The prediction and batch-job endpoints need the API key.
## Authentication

Forward team password using one of the two following headers:
```text
X-API-Key: YOUR_TEAM_API_KEY
```

or:

```text
Authorization: Bearer YOUR_TEAM_API_KEY
```

Keep the actual key secret. For a service that has already been deployed, add the key using the hosting environment variable settings.
## API endpoints

| Method | Endpoint | Purpose |
|---|---|---|
| `GET` | `/health` | Public health check |
| `POST` | `/predict` | Predict one ticket |
| `POST` | `/predict/batch` | Predict 1–100 tickets synchronously |
| `POST` | `/batch/jobs` | Submit a large batch job |
| `GET` | `/batch/jobs/{job_id}` | Check job status |
| `GET` | `/batch/jobs/{job_id}/results` | Retrieve completed job results |
| `DELETE` | `/batch/jobs/{job_id}` | Cancel or delete a job |

For big batch requests, there are more than 2,000 tickets and up to 5,000 tickets for each request. If the request is successful, the response will return a `202 Accepted` code and a `job_id`. The status endpoint should be polled until the job is successfully completed. Then the results can be fetched.
Every prediction and job endpoint requires authentication. All errors are in a JSON object with a `code` and `message`, and field information for any validation errors.
Authoritative Request and Response Contract is located at:
- `API specs and schemas/tensorforge-phase2-openapi-v2.yaml`
- `API specs and schemas/`

The auto-generated `/docs` page may not contain all of the request schemas or the authentication requirements since the API validates requests manually. Use the provided OpenAPI file and the schemas for the full contract.

## Example single-ticket request

```json
{
  "channel": "email",
  "subject": "Refund not received",
  "text": "I was charged twice and need a refund.",
  "ticket_id": "demo-001"
}
```

Allowed `channel` fields include `email`, `chat`, and `call_transcript`. The ticket text should include at least some non-whitespace text and shouldn’t exceed 10,000 characters. Subject field shouldn’t exceed 500 characters; ticket_id field shouldn’t exceed 64 characters.
## Test a prediction in PowerShell

With the API running in Docker, run:

```powershell
$apiKey = Read-Host "Enter the team API key"
$headers = @{ "X-API-Key" = $apiKey }
$body = @{
    channel = "email"
    subject = "Refund not received"
    text = "I was charged twice and need a refund."
    ticket_id = "demo-001"
} | ConvertTo-Json

Invoke-RestMethod `
    -Uri "http://127.0.0.1:8000/predict" `
    -Method Post `
    -Headers $headers `
    -ContentType "application/json" `
    -Body $body
```

## Locally run the browser demo

1. Begin the Docker API with the above command.
2. Open frontend/main.html in VS Code using the Live Server extension..
3. Ensure that the page is being loaded from `http://127.0.0.1:5500/` or `http://localhost:5500/` and not as a local file.
4. Enter a ticket channel, ticket subject, ticket text, and valid API key and click on **Predict Category**.

For the hosted frontend, make sure that the frontend is pointing to the API hosted in the server at `http://127.0.0.0.1:8000/predict` and add the actual frontend origin to the allow list in the backend. The API key should not be hardcoded in the browser JavaScript code.
## Model and validation results

The model employs the TF-IDF features of characters and Logistic Regression algorithm. It was trained on 4,000 tickets and tested on 800 validation tickets.
| Metric | Result |
|---|---:|
| Main-category accuracy | 58.63% |
| Main-category macro F1 | 0.609 |
| Urgency accuracy | 96.12% |
| Secondary-category accuracy, all tickets | 95.50% |
| Secondary accuracy when present | 55.00% |
| Recorded response-schema checks | 800/800 |

The general score for the secondary category classification is high since the validation tickets mostly do not have any secondary category. For those tickets that had another category, the score is 55.00%. The urgency classifier got 15 out of 80 urgent tickets wrong. Confidence is a model prediction probability that is not calibrated to certainty. This dataset is a synthetic one.
## Local deployment status

Both the API and browser demo were tested locally. Public hosting will be done soon. Provide the links to the live API and demo after deployment.
## Security and operational notes

- NEVER upload `.env` nor share the team API key. `.env` file is excluded via `.gitignore`. It is ignored from the Docker image via `.dockerignore`.
- The Docker image contains all model files, so the inference process does not require an internet connection.
- The batch-job data is stored in memory and gets destroyed once the API process is stopped
- This API key in the frontend is meant for the purposes of demo only. A shared secret should never be used in frontend JavaScript code.
