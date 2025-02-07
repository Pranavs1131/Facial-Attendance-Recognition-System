# Facial Attendance Recognition System

A facial recognition-based attendance system built using Flask, OpenCV, and machine learning algorithms. This application captures images, detects faces, and records attendance by recognizing individuals' faces.

## Features

- **Face Detection**: Utilizes OpenCV's Haar Cascade Classifier to detect faces in real-time.
- **Face Recognition**: Implements machine learning algorithms to recognize and differentiate between faces.
- **Attendance Logging**: Records attendance with timestamps upon successful face recognition.
- **Web Interface**: Provides a user-friendly interface using Flask for interaction.

## Prerequisites

Ensure you have the following installed:

- Python 3.x
- Flask
- OpenCV
- NumPy
- Pandas

You can install the required Python packages using:

```bash
pip install flask opencv-python numpy pandas
Installation
Clone the Repository:

bash
Copy
Edit
git clone https://github.com/Pranavs1131/Facial-Attendance-Recognition-System.git
cd Facial-Attendance-Recognition-System
Prepare the Dataset:

Place the images of individuals in the Attendance directory. Each image should be named with the person's ID and name (e.g., 1_JohnDoe.jpg).
Run the Application:

bash
Copy
Edit
python app2.py
The application will start, and you can access it by navigating to http://127.0.0.1:5000/ in your web browser.

Usage
Capture Images: Use the web interface to capture images for new individuals to add to the dataset.
Train the Model: After capturing images, train the model to recognize the new faces.
Mark Attendance: The system will automatically mark attendance when it recognizes a face from the dataset.
Contributing
Contributions are welcome! Please fork the repository and submit a pull request with your enhancements or bug fixes.
