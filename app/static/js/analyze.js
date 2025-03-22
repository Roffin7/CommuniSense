function uploadVideo() {
    let formData = new FormData();
    let fileInput = document.getElementById("videoFile");
    formData.append("video", fileInput.files[0]);

    showLoading();

    fetch("/upload", {
        method: "POST",
        body: formData
    })
    .then(response => response.json())
    .then(data => {
        hideLoading();
        displayResults(data);
        fetchExistingFiles();  // Refresh file list after upload
    })
    .catch(error => {
        hideLoading();
        console.error("Error:", error);
    });
}

function fetchExistingFiles() {
    fetch("/list_files")
    .then(response => response.json())
    .then(files => {
        let dropdown = document.getElementById("existingFiles");
        dropdown.innerHTML = "<option value=''>Select a file</option>";
        files.forEach(file => {
            let option = document.createElement("option");
            option.value = file;
            option.textContent = file;
            dropdown.appendChild(option);
        });
    })
    .catch(error => console.error("Error fetching files:", error));
}

function analyzeExistingFile() {
    let dropdown = document.getElementById("existingFiles");
    let selectedFile = dropdown.value;
    if (!selectedFile) {
        alert("Please select a file to analyze.");
        return;
    }

    showLoading();

    fetch("/analyze_existing", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ filename: selectedFile })
    })
    .then(response => response.json())
    .then(data => {
        hideLoading();
        displayResults(data);
    })
    .catch(error => {
        hideLoading();
        console.error("Error:", error);
    });
}

function displayResults(data) {
    document.getElementById("faceResult").innerText = data.face_detection;
    document.getElementById("clarityScore").innerText = data.voice_clarity_score + "/100";
    document.getElementById("speakingScore").innerText = data.overall_speaking_score;
    document.getElementById("feedback").innerText = "Feedback: " + data.final_feedback;
    document.getElementById("speakingPace").innerText = (data.speaking_pace_wpm || "N/A");
}

// Show loading animation
function showLoading() {
    document.getElementById("loading").classList.remove("hidden");
}

// Hide loading animation
function hideLoading() {
    document.getElementById("loading").classList.add("hidden");
}

window.onload = fetchExistingFiles;
