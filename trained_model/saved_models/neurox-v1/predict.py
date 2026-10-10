from pathlib import Path
import json
import joblib
import pandas as pd

# Find models.joblib beside this Python file.

model_path = Path(__file__).resolve().parent / "models.joblib"

# Load the saved models once
bundle = joblib.load(model_path)

category_model = bundle["category_model"]
urgency_model = bundle["urgency_model"]
secondary_model = bundle["secondary_model"]

TEAM_MAP = bundle["team_map"]
MODEL_VERSION = bundle["model_version"]
NO_SECONDARY = bundle["no_secondary_label"]


def make_ticket_text(data):
    # Use exactly the same text preparation as the training notebook
    return (
        data["channel"].astype(str)
        + " [SUBJECT] " + data["subject"].fillna("").astype(str)
        + " [TICKET] " + data["text"].astype(str)
    )


def predict_ticket(channel, text, subject="", ticket_id=None):
    # Check the ticket inputs
    if channel not in ("email", "chat", "call_transcript"):
        raise ValueError("Invalid channel.")

    if not isinstance(text, str) or not text.strip() or len(text) > 10000:
        raise ValueError("Text must be nonempty and at most 10,000 characters.")

    if not isinstance(subject, str) or len(subject) > 500:
        raise ValueError("Subject must be text, at most 500 characters.")

    if ticket_id is not None:
        if not isinstance(ticket_id, str) or len(ticket_id) > 64:
            raise ValueError("Ticket ID must be text, at most 64 characters.")

    # Prepare one ticket for the models
    ticket = pd.DataFrame([{
        "channel": channel,
        "subject": subject,
        "text": text
    }])

    ticket_text = make_ticket_text(ticket)

    # Predict the main category and get its probability estimate
    probabilities = category_model.predict_proba(ticket_text)[0]
    best_index = probabilities.argmax()

    category = str(category_model.classes_[best_index])
    confidence = float(probabilities[best_index])

    # Predict the secondary category and urgency
    secondary = str(secondary_model.predict(ticket_text)[0])
    urgent = bool(urgency_model.predict(ticket_text)[0])

    # Apply the competition's response rules
    if secondary == NO_SECONDARY or secondary == category:
        secondary = None

    if category == "spam_irrelevant":
        secondary = None
        urgent = False

    # Return the complete prediction
    result = {
        "category": category,
        "secondary_category": secondary,
        "team": TEAM_MAP[category],
        "is_urgent": urgent,
        "confidence": confidence,
        "model_version": MODEL_VERSION
    }

    if ticket_id is not None:
        result["ticket_id"] = ticket_id

    return result


# Run this example only when you run predict.py directly.
# It will not run when your backend imports predict_ticket.
if __name__ == "__main__":
    result = predict_ticket(
        channel="chat",
        text="I was charged twice for the same ride.",
        ticket_id="demo-001"
    )

    # Display the result neatly
    print(json.dumps(result, indent=2, ensure_ascii=False))