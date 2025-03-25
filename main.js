document.addEventListener("DOMContentLoaded", function () {
  initializeEventListeners();
});

function initializeEventListeners() {
  document
    .getElementById("userInputForm")
    .addEventListener("submit", handleUserInput);
  document
    .getElementById("excelUploadForm")
    .addEventListener("submit", handleExcelUpload);
}

function handleSelection() {
  const inputType = document.getElementById("inputType").value;
  document.getElementById("userInputSection").style.display =
    inputType === "userInput" ? "block" : "none";
  document.getElementById("excelUploadSection").style.display =
    inputType === "excelUpload" ? "block" : "none";
  document.getElementById("resultsSection").style.display = "none";
}

function handleProgramSelection() {
  const programName = document.getElementById("programName").value;
  const dynamicFields = document.getElementById("dynamicFields");

  dynamicFields.innerHTML = "";

  if (programName === "SelectionGapClosure") {
    dynamicFields.innerHTML = `
          <div class="form-group">
              <label for="application">Select Application:</label>
              <select id="application" name="application" required>
                  <option value="EBHS">EBHS</option>
                  <option value="EBS">EBS</option>
                  <option value="EmBHS">EmBHS</option>
                  <option value="DAWN">DAWN</option>
              </select>
          </div>
          <div class="form-group">
              <label for="sourceDqCleared">Is source DQ cleared?</label>
              <select id="sourceDqCleared" name="sourceDqCleared" required>
                  <option value="true">true</option>
                  <option value="false">false</option>
              </select>
          </div>
          <div class="form-group">
              <label for="matchingModel">Select Matching Model:</label>
              <select id="matchingModel" name="matchingModel" required>
                  <option value="epm1dot1tt">epm1dot1tt</option>
                  <option value="epm1dot1">epm1dot1</option>
                  <option value="epm1dot1small">epm1dot1small</option>
                  <option value="epmbasevariant">epmbasevariant</option>
                  <option value="epmbaseexact">epmbaseexact</option>
                  <option value="vibe">vibe</option>
              </select>
          </div>
          <div class="form-group">
              <label for="dedupedPipeline">Is deduped pipeline enabled?</label>
              <select id="dedupedPipeline" name="dedupedPipeline" required>
                  <option value="true">true</option>
                  <option value="false">false</option>
              </select>
          </div>
      `;
  }
}

async function handleUserInput(event) {
  event.preventDefault();
  showLoading();

  try {
    const formData = new FormData(event.target);
    const response = await fetch("/submit_user_input", {
      method: "POST",
      body: formData,
    });

    const result = await response.json();
    displayResults(result);
  } catch (error) {
    displayError("An error occurred while processing your request");
    console.error("Error:", error);
  } finally {
    hideLoading();
  }
}

async function handleExcelUpload(event) {
  event.preventDefault();
  showLoading();

  try {
    const formData = new FormData(event.target);
    const response = await fetch("/submit_excel", {
      method: "POST",
      body: formData,
    });

    const result = await response.json();
    displayResults(result);
  } catch (error) {
    displayError("An error occurred while uploading files");
    console.error("Error:", error);
  } finally {
    hideLoading();
  }
}

function displayResults(result) {
  const resultsSection = document.getElementById("resultsSection");
  const resultMessages = document.getElementById("resultMessages");
  const downloadLinks = document.getElementById("downloadLinks");

  resultMessages.innerHTML = "";
  downloadLinks.innerHTML = "";

  if (result.status === "success") {
    // Display messages
    result.messages.forEach((message) => {
      const messageElement = document.createElement("p");
      messageElement.textContent = message;
      resultMessages.appendChild(messageElement);
    });

    // Add download links
    if (result.downloads) {
      result.downloads.forEach((download) => {
        const link = document.createElement("a");
        link.href = download.url;
        link.className = "download-link";
        link.textContent = `Download ${download.filename}`;
        downloadLinks.appendChild(link);
      });
    }
  } else {
    displayError(result.message);
  }

  resultsSection.style.display = "block";
}

function displayError(message) {
  const resultsSection = document.getElementById("resultsSection");
  const resultMessages = document.getElementById("resultMessages");

  resultMessages.innerHTML = `
      <div class="error-message">
          ${message}
      </div>
  `;
  resultsSection.style.display = "block";
}

function showLoading() {
  document.getElementById("loadingSpinner").style.display = "block";
}

function hideLoading() {
  document.getElementById("loadingSpinner").style.display = "none";
}
