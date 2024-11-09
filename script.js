function showSection(section) {
    // Hide all sections
    document.querySelectorAll('.section').forEach((el) => {
        el.style.display = 'none';
    });
    
    // Show the selected section
    document.getElementById(section).style.display = 'block';
}

function showPopup(detectionType) {
    document.getElementById('popup-title').innerText = detectionType;
    document.getElementById('popup').style.display = 'block';
    document.getElementById('result').innerHTML = '';  // Clear previous result
}

function closePopup() {
    document.getElementById('popup').style.display = 'none';
    document.getElementById('camera').style.display = 'none';
    document.getElementById('captureBtn').style.display = 'none';
    document.getElementById('result').innerHTML = '';  // Clear result on close
}

function uploadImage() {
    const input = document.createElement('input');
    input.type = 'file';
    input.accept = 'image/*';
    input.onchange = (event) => {
        const file = event.target.files[0];
        if (file) {
            sendImageToAPI(file);
        }
    };
    input.click();
}

function captureImage() {
    const video = document.getElementById('camera');
    const captureBtn = document.getElementById('captureBtn');
    
    video.style.display = 'block';
    captureBtn.style.display = 'inline-block';

    // Access user's camera
    navigator.mediaDevices.getUserMedia({ video: true })
        .then((stream) => {
            video.srcObject = stream;
            window.localStream = stream; // Save the stream to stop later
        })
        .catch((err) => {
            console.error("Error accessing the camera:", err);
        });
}

function captureAndSaveImage() {
    const video = document.getElementById('camera');
    const canvas = document.createElement('canvas');
    const context = canvas.getContext('2d');
    
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    context.drawImage(video, 0, 0, canvas.width, canvas.height);

    // Convert canvas to blob
    canvas.toBlob((blob) => {
        sendImageToAPI(blob);

        // Stop the camera after capture
        const tracks = window.localStream.getTracks();
        tracks.forEach(track => track.stop());
        video.style.display = 'none';
        document.getElementById('captureBtn').style.display = 'none';
    });
}

function sendImageToAPI(image) {
    const formData = new FormData();
    formData.append('file', image);

    // Send the image to the backend Flask API
    fetch('http://localhost:5000/predict', {  // Replace with your actual API URL
        method: 'POST',
        body: formData,
    })
    .then(response => response.json())
    .then(data => {
        // Display the result
        document.getElementById('result').innerHTML = `Prediction: ${data.prediction}, Confidence: ${data.confidence.toFixed(2)}`;
    })
    .catch(error => {
        console.error('Error:', error);
        document.getElementById('result').innerHTML = 'Error processing the image.';
    });
}
