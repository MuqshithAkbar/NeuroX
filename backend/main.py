import threading
import uuid
from datetime import datetime, timedelta, timezone

import hmac
import json
import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from fastapi import BackgroundTasks, FastAPI, Request, Response
from fastapi.responses import JSONResponse
from jsonschema import Draft202012Validator

# Find the main NeuroX folder and load its local API key from .env.
PROJECT_FOLDER = Path(__file__).resolve().parents[1]
load_dotenv(PROJECT_FOLDER / ".env")

# Locate the saved model and its prediction function.
MODEL_FOLDER = (
    PROJECT_FOLDER
    / "trained_model"
    / "saved_models"
    / "neurox-v1"
)
sys.path.insert(0, str(MODEL_FOLDER))

# Importing predict.py loads the saved model.
from predict import MODEL_VERSION, predict_ticket  # noqa: E402

# Load the official schema used to validate one ticket.
SCHEMA_FILE = (
    PROJECT_FOLDER
    / "API specs and schemas"
    / "predict_request.schema.json"
)
with SCHEMA_FILE.open(encoding="utf-8") as schema_file:
    REQUEST_SCHEMA = json.load(schema_file)

REQUEST_VALIDATOR = Draft202012Validator(REQUEST_SCHEMA)

# Load the official schema for batches of up to 100 tickets.
BATCH_SCHEMA_FILE = (
    PROJECT_FOLDER
    / "API specs and schemas"
    / "batch_request.schema.json"
)
with BATCH_SCHEMA_FILE.open(encoding="utf-8") as schema_file:
    BATCH_SCHEMA = json.load(schema_file)

BATCH_VALIDATOR = Draft202012Validator(BATCH_SCHEMA)

# Load the official schema for large batch jobs.
JOB_SCHEMA_FILE = (
    PROJECT_FOLDER
    / "API specs and schemas"
    / "batch_job_request.schema.json"
)
with JOB_SCHEMA_FILE.open(encoding="utf-8") as schema_file:
    JOB_SCHEMA = json.load(schema_file)

JOB_VALIDATOR = Draft202012Validator(JOB_SCHEMA)

# Keep active job data in memory for this first implementation.
JOBS = {}
JOBS_LOCK = threading.Lock()
IDEMPOTENCY_JOBS = {}
MAX_ACTIVE_JOBS = 5
MAX_JOB_BODY_BYTES = 25 * 1024 * 1024

MAX_BODY_BYTES = 5 * 1024 * 1024

# Create the API application.
app = FastAPI(title="NeuroX Ticket Classifier")


def make_error(status_code, code, message, details=None, headers=None):
    # Return errors in the format required by the competition.
    error = {"code": code, "message": message}
    if details is not None:
        error["details"] = details

    return JSONResponse(
        status_code=status_code,
        content={"error": error},
        headers=headers,
    )


def has_valid_api_key(request):
    # Accept either X-API-Key or Authorization: Bearer.
    candidates = []

    api_key_header = request.headers.get("x-api-key")
    if api_key_header:
        candidates.append(api_key_header)

    authorization = request.headers.get("authorization", "")
    scheme, separator, token = authorization.partition(" ")
    if separator and scheme.lower() == "bearer" and token:
        candidates.append(token)

    expected_key = os.environ.get("API_KEY")

    # Refuse access if no key was loaded from .env or the environment.
    if not expected_key:
        return False

    # Compare securely without printing or logging either key.
    return any(
        hmac.compare_digest(candidate, expected_key)
        for candidate in candidates
    )


@app.get("/health")
def health():
    # This endpoint is public. Successful model import means it is loaded.
    return {
        "status": "ok",
        "model_version": MODEL_VERSION,
        "model_loaded": True,
    }


@app.post("/predict")
async def predict_one(request: Request):
    # Check the API key before reading or parsing the request body.
    if not has_valid_api_key(request):
        return make_error(
            401,
            "unauthorized",
            "A valid API key is required.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    # Require JSON content.
    content_type = request.headers.get("content-type", "")
    media_type = content_type.split(";", 1)[0].strip().lower()
    if media_type != "application/json":
        return make_error(
            415,
            "unsupported_media_type",
            "Content-Type must be application/json.",
        )

    # Enforce the request-size limit.
    content_length = request.headers.get("content-length")
    if content_length and content_length.isdigit():
        if int(content_length) > MAX_BODY_BYTES:
            return make_error(
                413,
                "payload_too_large",
                "Request body exceeds the 5 MB limit.",
            )

    raw_body = await request.body()
    if len(raw_body) > MAX_BODY_BYTES:
        return make_error(
            413,
            "payload_too_large",
            "Request body exceeds the 5 MB limit.",
        )

    # Parse JSON and return a clear error if it is malformed.
    try:
        ticket = json.loads(raw_body)
    except (UnicodeDecodeError, json.JSONDecodeError):
        return make_error(400, "invalid_json", "Request body must be valid JSON.")

    # Check the ticket fields and limits against the official schema.
    validation_errors = sorted(
        REQUEST_VALIDATOR.iter_errors(ticket),
        key=lambda error: list(map(str, error.absolute_path)),
    )
    if validation_errors:
        details = []
        for error in validation_errors:
            field = ".".join(str(part) for part in error.absolute_path) or "body"
            details.append({"field": field, "issue": error.message})

        return make_error(
            422,
            "validation_error",
            "Request does not match the ticket schema.",
            details,
        )

    # Send the accepted fields to your saved model.
    return predict_ticket(
        channel=ticket["channel"],
        subject=ticket.get("subject", ""),
        text=ticket["text"],
        ticket_id=ticket.get("ticket_id"),
    )

@app.post("/predict/batch")
async def predict_batch(request: Request):
    # Check the API key before reading or parsing the body.
    if not has_valid_api_key(request):
        return make_error(
            401,
            "unauthorized",
            "A valid API key is required.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    # Require JSON content.
    content_type = request.headers.get("content-type", "")
    media_type = content_type.split(";", 1)[0].strip().lower()
    if media_type != "application/json":
        return make_error(
            415,
            "unsupported_media_type",
            "Content-Type must be application/json.",
        )

    # Enforce the 5 MB request limit.
    content_length = request.headers.get("content-length")
    if content_length and content_length.isdigit():
        if int(content_length) > MAX_BODY_BYTES:
            return make_error(
                413,
                "payload_too_large",
                "Request body exceeds the 5 MB limit.",
            )

    raw_body = await request.body()
    if len(raw_body) > MAX_BODY_BYTES:
        return make_error(
            413,
            "payload_too_large",
            "Request body exceeds the 5 MB limit.",
        )

    # Parse the JSON body.
    try:
        batch = json.loads(raw_body)
    except (UnicodeDecodeError, json.JSONDecodeError):
        return make_error(400, "invalid_json", "Request body must be valid JSON.")

    # Validate the whole batch; no predictions are made if any ticket is invalid.
    validation_errors = list(BATCH_VALIDATOR.iter_errors(batch))
    if validation_errors:
        details = []

        for error in validation_errors:
            path = list(error.absolute_path)

            # The ticket index is the number after "tickets" in the schema path.
            index = None
            if (
                len(path) > 1
                and path[0] == "tickets"
                and isinstance(path[1], int)
            ):
                index = path[1]
                field_path = path[2:]
            else:
                field_path = path

            field = ".".join(str(part) for part in field_path) or "tickets"
            detail = {"field": field, "issue": error.message}

            # Include the failing ticket's zero-based position when available.
            if index is not None:
                detail["index"] = index

            details.append(detail)

        return make_error(
            422,
            "validation_error",
            "One or more tickets do not match the batch schema.",
            details,
        )

    tickets = batch["tickets"]

    # The schema can't check duplicate IDs, so check them here.
    seen_ticket_ids = set()
    duplicate_details = []

    for index, ticket in enumerate(tickets):
        ticket_id = ticket["ticket_id"]

        if ticket_id in seen_ticket_ids:
            duplicate_details.append({
                "index": index,
                "field": "ticket_id",
                "issue": "must be unique within the batch",
            })
        else:
            seen_ticket_ids.add(ticket_id)

    if duplicate_details:
        return make_error(
            422,
            "validation_error",
            "Ticket IDs must be unique within the batch.",
            duplicate_details,
        )

    # Predict each ticket independently and keep the submitted order.
    predictions = [
        predict_ticket(
            channel=ticket["channel"],
            subject=ticket.get("subject", ""),
            text=ticket["text"],
            ticket_id=ticket["ticket_id"],
        )
        for ticket in tickets
    ]

    # Return one prediction per ticket, in the same order as the request.
    return {
        "predictions": predictions,
        "meta": {
            "count": len(predictions),
            "model_version": MODEL_VERSION,
        },
    }

def utc_now():
    # Format timestamps as UTC for the job status responses.
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def job_status(job):
    # Return only the fields used by the job status API.
    return {
        "job_id": job["job_id"],
        "status": job["status"],
        "total": job["total"],
        "processed": job["processed"],
        "created_at": job["created_at"],
        "started_at": job["started_at"],
        "finished_at": job["finished_at"],
        "expires_at": job["expires_at"],
        "model_version": MODEL_VERSION,
        "error": job["error"],
    }


def run_batch_job(job_id, tickets):
    # Mark the accepted job as running.
    with JOBS_LOCK:
        job = JOBS.get(job_id)
        if job is None or job["status"] == "cancelled":
            return

        job["status"] = "running"
        job["started_at"] = utc_now()

    try:
        # Classify each ticket and update progress as it finishes.
        for index, ticket in enumerate(tickets):
            with JOBS_LOCK:
                job = JOBS.get(job_id)
                if job is None or job["status"] == "cancelled":
                    return

            prediction = predict_ticket(
                channel=ticket["channel"],
                subject=ticket.get("subject", ""),
                text=ticket["text"],
                ticket_id=ticket["ticket_id"],
            )

            with JOBS_LOCK:
                job = JOBS.get(job_id)
                if job is None or job["status"] == "cancelled":
                    return

                job["predictions"].append(prediction)
                job["processed"] = index + 1

        # Mark the job complete and retain its results for at least six hours.
        with JOBS_LOCK:
            job = JOBS.get(job_id)
            if job is not None and job["status"] != "cancelled":
                job["status"] = "succeeded"
                job["finished_at"] = utc_now()
                job["expires_at"] = (
                    datetime.now(timezone.utc) + timedelta(hours=6)
                ).isoformat().replace("+00:00", "Z")

    except Exception:
        # Do not expose internal exception details in the API response.
        with JOBS_LOCK:
            job = JOBS.get(job_id)
            if job is not None and job["status"] != "cancelled":
                job["status"] = "failed"
                job["finished_at"] = utc_now()
                job["expires_at"] = (
                    datetime.now(timezone.utc) + timedelta(hours=6)
                ).isoformat().replace("+00:00", "Z")
                job["error"] = {
                    "code": "processing_error",
                    "message": "The batch job could not be completed.",
                }


@app.post("/batch/jobs")
async def submit_batch_job(
    request: Request,
    background_tasks: BackgroundTasks,
):
    # Check the API key before reading or parsing the body.
    if not has_valid_api_key(request):
        return make_error(
            401,
            "unauthorized",
            "A valid API key is required.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    # Require JSON content.
    content_type = request.headers.get("content-type", "")
    media_type = content_type.split(";", 1)[0].strip().lower()
    if media_type != "application/json":
        return make_error(
            415,
            "unsupported_media_type",
            "Content-Type must be application/json.",
        )

    # Enforce the 25 MB limit for large-job requests.
    content_length = request.headers.get("content-length")
    if content_length and content_length.isdigit():
        if int(content_length) > MAX_JOB_BODY_BYTES:
            return make_error(
                413,
                "payload_too_large",
                "Request body exceeds the 25 MB limit.",
            )

    raw_body = await request.body()
    if len(raw_body) > MAX_JOB_BODY_BYTES:
        return make_error(
            413,
            "payload_too_large",
            "Request body exceeds the 25 MB limit.",
        )

    # Parse the submitted JSON.
    try:
        batch = json.loads(raw_body)
    except (UnicodeDecodeError, json.JSONDecodeError):
        return make_error(400, "invalid_json", "Request body must be valid JSON.")

    # Validate every ticket before creating a job.
    validation_errors = list(JOB_VALIDATOR.iter_errors(batch))
    if validation_errors:
        details = []

        for error in validation_errors:
            path = list(error.absolute_path)
            index = None

            # Identify which ticket caused the schema error, if possible.
            if (
                len(path) > 1
                and path[0] == "tickets"
                and isinstance(path[1], int)
            ):
                index = path[1]
                field_path = path[2:]
            else:
                field_path = path

            field = ".".join(str(part) for part in field_path) or "tickets"
            detail = {"field": field, "issue": error.message}

            if index is not None:
                detail["index"] = index

            details.append(detail)

        return make_error(
            422,
            "validation_error",
            "One or more tickets do not match the job schema.",
            details,
        )

    tickets = batch["tickets"]

    # Ticket IDs must be unique within this job.
    seen_ticket_ids = set()
    duplicate_details = []

    for index, ticket in enumerate(tickets):
        ticket_id = ticket["ticket_id"]

        if ticket_id in seen_ticket_ids:
            duplicate_details.append({
                "index": index,
                "field": "ticket_id",
                "issue": "must be unique within the job",
            })
        else:
            seen_ticket_ids.add(ticket_id)

    if duplicate_details:
        return make_error(
            422,
            "validation_error",
            "Ticket IDs must be unique within the job.",
            duplicate_details,
        )

    # Reuse the existing job if the caller retries with the same idempotency key.
    idempotency_key = request.headers.get("idempotency-key")
    with JOBS_LOCK:
        if idempotency_key:
            existing_id = IDEMPOTENCY_JOBS.get(idempotency_key)
            if existing_id and existing_id in JOBS:
                existing_job = JOBS[existing_id]
                return JSONResponse(
                    status_code=202,
                    content=job_status(existing_job),
                    headers={
                        "Location": f"/batch/jobs/{existing_id}",
                        "Retry-After": "2",
                    },
                )

        # Limit how many jobs can wait or run at once.
        active_count = sum(
            job["status"] in ("queued", "running")
            for job in JOBS.values()
        )
        if active_count >= MAX_ACTIVE_JOBS:
            return make_error(
                429,
                "too_many_jobs",
                "The service is busy. Please retry later.",
                headers={"Retry-After": "5"},
            )

        # Create the queued job record.
        job_id = str(uuid.uuid4())
        job = {
            "job_id": job_id,
            "status": "queued",
            "total": len(tickets),
            "processed": 0,
            "created_at": utc_now(),
            "started_at": None,
            "finished_at": None,
            "expires_at": None,
            "predictions": [],
            "error": None,
        }
        JOBS[job_id] = job

        if idempotency_key:
            IDEMPOTENCY_JOBS[idempotency_key] = job_id

    # Start classifying after returning the accepted-job response.
    background_tasks.add_task(run_batch_job, job_id, tickets)

    return JSONResponse(
        status_code=202,
        content=job_status(job),
        headers={
            "Location": f"/batch/jobs/{job_id}",
            "Retry-After": "2",
        },
    )

def job_has_expired(job):
    # Results expire at the time recorded when the job finishes.
    expires_at = job.get("expires_at")
    if not expires_at:
        return False

    expiry_time = datetime.fromisoformat(expires_at.replace("Z", "+00:00"))
    expired = datetime.now(timezone.utc) >= expiry_time

    # Discard predictions after their retention period.
    if expired:
        job["predictions"].clear()

    return expired


@app.get("/batch/jobs/{job_id}")
def get_batch_job(job_id: str, request: Request):
    # Protect job status with the same API key as prediction routes.
    if not has_valid_api_key(request):
        return make_error(
            401,
            "unauthorized",
            "A valid API key is required.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    # Find the job and copy its status while holding the lock.
    with JOBS_LOCK:
        job = JOBS.get(job_id)
        if job is None:
            return make_error(404, "job_not_found", "No job with this id.")

        if job_has_expired(job):
            expired = True
            status = None
        else:
            expired = False
            status = job_status(job)

    if expired:
        return make_error(410, "job_expired", "Job results have expired.")

    # Tell the caller when to poll again while the job is active.
    headers = {}
    if status["status"] in ("queued", "running"):
        headers["Retry-After"] = "2"

    return JSONResponse(content=status, headers=headers)


@app.get("/batch/jobs/{job_id}/results")
def get_batch_job_results(job_id: str, request: Request):
    # Protect results with the same API key as prediction routes.
    if not has_valid_api_key(request):
        return make_error(
            401,
            "unauthorized",
            "A valid API key is required.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    # Read and validate paging options from the URL.
    try:
        offset = int(request.query_params.get("offset", "0"))
        limit_text = request.query_params.get("limit")
        limit = int(limit_text) if limit_text is not None else None
    except ValueError:
        return make_error(
            422,
            "validation_error",
            "Offset and limit must be whole numbers.",
            [
                {
                    "field": "offset or limit",
                    "issue": "must be a whole number",
                }
            ],
        )

    # Enforce the paging limits from the API contract.
    if offset < 0:
        return make_error(
            422,
            "validation_error",
            "Offset must be zero or greater.",
            [{"field": "offset", "issue": "must be zero or greater"}],
        )

    if limit is not None and not 1 <= limit <= 5000:
        return make_error(
            422,
            "validation_error",
            "Limit must be between 1 and 5,000.",
            [{"field": "limit", "issue": "must be between 1 and 5,000"}],
        )

    # Find the job and copy its results while holding the lock.
    with JOBS_LOCK:
        job = JOBS.get(job_id)
        if job is None:
            return make_error(404, "job_not_found", "No job with this id.")

        if job_has_expired(job):
            expired = True
            result = None
        elif job["status"] != "succeeded":
            expired = False
            not_ready = job["status"]
            result = None
        else:
            expired = False
            not_ready = None

            # When limit is omitted, return the whole result.
            page_limit = limit if limit is not None else job["total"]
            page = job["predictions"][offset : offset + page_limit]
            next_offset = offset + len(page)

            result = {
                "job_id": job_id,
                "status": "succeeded",
                "total": job["total"],
                "offset": offset,
                "limit": page_limit,
                "next_offset": (
                    next_offset if next_offset < job["total"] else None
                ),
                "model_version": MODEL_VERSION,
                "predictions": page,
            }

    if expired:
        return make_error(410, "job_expired", "Job results have expired.")

    if not_ready is not None:
        return make_error(
            409,
            "job_not_ready",
            f"Job status is '{not_ready}'. Results are available once it is 'succeeded'.",
        )

    return result


@app.delete("/batch/jobs/{job_id}", status_code=204)
def delete_batch_job(job_id: str, request: Request):
    # Protect cancellation and deletion with the same API key.
    if not has_valid_api_key(request):
        return make_error(
            401,
            "unauthorized",
            "A valid API key is required.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    # Remove the job so it can no longer be polled or fetched.
    with JOBS_LOCK:
        job = JOBS.get(job_id)
        if job is None:
            return make_error(404, "job_not_found", "No job with this id.")

        job["status"] = "cancelled"
        job["predictions"].clear()
        del JOBS[job_id]

        # Remove any idempotency key that points to this job.
        keys_to_remove = [
            key
            for key, saved_job_id in IDEMPOTENCY_JOBS.items()
            if saved_job_id == job_id
        ]
        for key in keys_to_remove:
            del IDEMPOTENCY_JOBS[key]

    # The API contract requires an empty 204 response.
    return Response(status_code=204)