const form = document.getElementById("ticketForm");

form.addEventListener("submit", async function(event) {

    event.preventDefault();

    const title = document.getElementById("title").value;
    const description = document.getElementById("description").value;

    try {

        const response = await fetch("http://localhost:8000/predict", {

            method: "POST",

            headers: {
                "Content-Type": "application/json"
            },

            body: JSON.stringify({
                title: title,
                description: description
            })

        });

        const data = await response.json();

        document.getElementById("category").textContent =
            data.category;

        document.getElementById("team").textContent =
            data.team;

    } catch (error) {

        console.error(error);

        alert("Could not connect to the backend API.");

    }

});