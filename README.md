# E-Waste Intelligence Suite

An upgraded Streamlit application for classifying electronic waste images into 10 categories using a TensorFlow MobileNetV2 model.

## Features

- Local user login and account creation
- Saved per-user history stored on disk
- Single-image prediction with top-3 confidence ranking
- Batch image analysis for multiple files in one run
- CSV export for analyzed results
- Plotly analytics dashboard for prediction trends and confidence bands
- File upload, camera capture, and live webcam page
- Built-in retraining page for creating updated model variants
- Cached model loading for faster repeated inference
- Handling notes and disposal guidance for each predicted category
- Safer error handling and cleaner project presentation

## Supported Categories

- Battery
- Keyboard
- Microwave
- Mobile
- Mouse
- PCB
- Player
- Printer
- Television
- Washing Machine

## Project Structure

```text
e-waste-classification/
|-- app.py
|-- model/
|   `-- e-waste-model-MobileNetV2.keras
|-- requirements.txt
`-- README.md
```

## Tech Stack

- Python 3.x
- Streamlit
- TensorFlow / Keras
- NumPy
- Pillow
- Plotly
- streamlit-webrtc
- OpenCV

## Run Locally

1. Create and activate a virtual environment.

```bash
python -m venv venv
venv\Scripts\activate
```

2. Install the dependencies.

```bash
pip install -r requirements.txt
```

3. Start the application.

```bash
streamlit run app.py
```

4. Open the local Streamlit URL in your browser and upload an image.

## User Data

- User accounts are stored locally in `data/users.json`.
- Each user's prediction history is saved in `data/history/<username>.json`.
- This implementation is intended for local/demo use and does not provide production-grade authentication security.

## Retraining

- Create a labeled dataset folder where each class has its own subfolder of images.
- Open the `Retrain Model` tab in the app.
- Set the dataset folder path, choose training settings, and save a new `.keras` model variant.
- Retraining currently builds a fresh MobileNetV2-based classifier using the provided dataset structure.

## Notes

- The app expects the trained model file at `model/e-waste-model-MobileNetV2.keras`.
- Predictions are intended for educational and demonstration use.
- For real-world disposal, always follow local recycling and hazardous waste guidelines.
