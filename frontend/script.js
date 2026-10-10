// This address is for testing while the API runs on your own computer.
const API_URL = "http://127.0.0.1:8000/predict";

const form = document.getElementById("ticketForm");
const message = document.getElementById("message");
const button = document.getElementById("submitButton");

form.addEventListener("submit", async (event) => {
    // Keep the browser on this page when the form is submitted.
    event.preventDefault();

    const channel = document.getElementById("channel").value;
    const subject = document.getElementById("title").value.trim();
    const text = document.getElementById("description").value.trim();
    const apiKey = document.getElementById("apiKey").value.trim();

    message.textContent = "Getting prediction…";
    button.disabled = true;

    try {
        const response = await fetch(API_URL, {
            method: "POST",
            headers: {
                "Content-Type": "application/json",
                // The API checks this header before it accepts the request.
                "X-API-Key": apiKey
            },
            body: JSON.stringify({
                channel: channel,
                subject: subject,
                text: text
            })
        });

        const data = await response.json();

        // Show the API's error message, such as an invalid key or bad request.
        if (!response.ok) {
            const errorText = data.error?.message || `Request failed (${response.status})`;
            throw new Error(errorText);
        }

        // Show the prediction returned by the API.
        document.getElementById("category").textContent = data.category;
        document.getElementById("team").textContent = data.team;
        document.getElementById("urgent").textContent = data.is_urgent ? "Yes" : "No";
        document.getElementById("confidence").textContent =
            `${(data.confidence * 100).toFixed(1)}%`;

        document.getElementById("result").hidden = false;
        message.textContent = "Prediction complete.";
    } catch (error) {
        // This also shows a browser message if the API cannot be reached.
        message.textContent = `Could not get a prediction: ${error.message}`;
        document.getElementById("result").hidden = true;
    } finally {
        button.disabled = false;
    }
});