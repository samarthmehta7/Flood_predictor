document.addEventListener("DOMContentLoaded", function () {
  updateCityDropdown(); // Initialize the city dropdown on page load
 
  // Add an event listener for the state dropdown to update the city dropdown
  document.getElementById("location").addEventListener("change", updateCityDropdown);
});

function updateCityDropdown() {
  var stateDropdown = document.getElementById("location");
  var cityDropdown = document.getElementById("city");

  // Clear existing options
  cityDropdown.innerHTML = "";

  // Get the selected state
  var selectedState = stateDropdown.value;

  // If no state is selected, return
  if (!selectedState) {
    return;
  }

  // Define cities based on the selected state
  var cities = getCities(selectedState);

  // Populate the city dropdown with new options
  for (var i = 0; i < cities.length; i++) {
    var option = document.createElement("option");
    option.text = cities[i];
    cityDropdown.add(option);
  }
}

function getCities(selectedState) {
  switch (selectedState) {
    case "Haryana":
      return ["Sirsa", "Hisar", "Bhiwani","Jind","Fatehabad"];
    case "Punjab":
      return ["Faridkot","Ferozpur","Jalandhar","Patiala","Roopnagar"];
    // Add more cases as needed
    default:
      return [];
  }
}


function submitForm() {
  var locationValue = document.getElementById("location").value;
  var cityValue = document.getElementById("city").value;
  var dateInputValue = document.getElementById("dateInput").value;

  // Do something with the form data
  console.log("Location: " + location);
  console.log("city: " + city);
  console.log("Date: " + dateInput);

  window.location.href = 'output1.html?location=' + encodeURIComponent(locationValue) +
  '&city=' + encodeURIComponent(cityValue) +
  '&dateInput=' + encodeURIComponent(dateInputValue);
}





 // static/script.js
// script.js

document.addEventListener('DOMContentLoaded', function () {
  // Check the URL for mode information
  const urlParams = new URLSearchParams(window.location.search);
  const modeFromURL = urlParams.get('mode');

  // If mode information is present, apply it; otherwise, default to light mode
  if (modeFromURL) {
      setMode(modeFromURL);
  } else {
      setMode('light-mode');
  }

  // Add event listener for the mode toggle
  const modeToggle = document.getElementById('modeToggle');
  modeToggle.addEventListener('change', handleModeToggle);
});

function handleModeToggle() {
  const modeToggle = document.getElementById('modeToggle');
  const mode = modeToggle.checked ? 'dark-mode' : 'light-mode';

  // Save the mode in local storage
  localStorage.setItem('mode', mode);

  // Update the URL with the new mode information
  updateURLWithMode(mode);

  // Apply the selected mode
  setMode(mode);
}

function setMode(mode) {
  document.body.className = mode;
}

function updateURLWithMode(mode) {
  const urlParams = new URLSearchParams(window.location.search);

  // Update the mode parameter in the URL
  urlParams.set('mode', mode);

  // Replace the current URL with the updated one
  history.replaceState(null, '', window.location.pathname + '?' + urlParams.toString());
}
