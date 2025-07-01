# 🧠 E-Waste Image Classification Web App

This is an AI-powered image classification project that identifies 10 types of electronic waste using deep learning (TensorFlow) and Streamlit.

## 🚀 Features
- Classifies 10 e-waste categories:  
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
- Web app built with Streamlit
- Simple drag & drop image upload
- Shows predicted class with confidence score

## 📁 Project Structure
e-waste-classification/
├── app.py
├── model/
│ └── e-waste-model-10class.keras
├── requirements.txt
├── .gitignore
└── README.md


## 🧰 Tech Stack
- Python 3.10
- TensorFlow
- Streamlit
-  MobileNetV2 Architecture — for efficient transfer learning and image classification.
- NumPy
- Pillow
- Google Colab (for training)
- VS Code (for web app deployment)

## 🔧 How to Run the Project

### 1. Clone the repository
```bash
git clone https://github.com/your-username/e-waste-classification.git
cd e-waste-classification

2. Create and activate virtual environment
python -m venv venv
venv\Scripts\activate  # On Windows

3. Install required packages
pip install -r requirements.txt

4. Start the Streamlit web app
streamlit run app.py


📜 License
This project is for educational purposes. Feel free to use and modify it.
