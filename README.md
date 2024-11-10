
# 🌿 HemoVeda: AI-Powered Anemia Detection Web App 🌿

Welcome to HemoVeda, a cutting-edge AI-driven solution designed to assist in early anemia detection. HemoVeda uses advanced deep learning models to analyze images of palms and conjunctiva (eye region) alongside blood report data to provide an accurate and non-invasive assessment of anemia risk. Our mission is to empower individuals and healthcare providers to act swiftly, preventing the serious health complications that untreated anemia can cause.
## 🩺 Problem Statement
Anemia affects over 50% of Indian women, yet many cases remain undiagnosed due to limited access to healthcare and diagnostic resources. Delayed detection of anemia can lead to severe health issues, including fatigue, pregnancy complications, and long-term physical challenges.

In regions where healthcare resources are scarce, a non-invasive, accessible approach to identifying anemia risk is essential for early intervention.


## 💡 Our Proposed Solution
Anemia is characterized by a deficiency in red blood cells (RBCs) or hemoglobin, which impacts oxygen circulation and can visibly affect certain parts of the body, including:

- Skin and Mucous Membranes
- Nails
- Tongue
- Palms and Conjunctiva

We focus on palm and conjunctiva imaging, as these areas often display early signs of anemia (e.g., paleness due to low hemoglobin levels) and are relatively easy to capture using non-invasive imaging methods.

Key Features:
- Image Analysis for Anemia Detection: 
     Using a deep learning-based model, HemoVeda analyzes visual cues in conjunctiva and palm images, detecting anemia-related changes in blood oxygenation and circulation.

- Multi-Input Model with Blood Data Integration"
    For improved accuracy, HemoVeda incorporates blood test data into its predictions, combining visual analysis with clinical data to offer a comprehensive assessment.

Empirical Model Testing
A range of deep learning (DL) models is tested and evaluated to identify the most effective architecture for this multi-input analysis.
## 🚀Features

- AI-Driven Detection: Advanced neural networks analyze conjunctiva and palm images to identify anemia symptoms.
- Multi-Input Integration: Combines blood report data with visual image analysis for a holistic and accurate diagnosis.
- Non-Invasive & Accessible: Ideal for populations lacking regular access to clinical facilities.
- Instant Results: Upload images and input blood data to receive an immediate anemia risk assessment.


## 📸Dataset
The HemoVeda model is trained on a specialized dataset that includes:

- Conjunctiva and Palm Images: Labeled as indicative of anemic or healthy conditions.
- Blood Report Data: Blood test information linked to each individual to supplement image-based predictions.

      Data sources: 1. Asare, Justice Williams; APPIAHENE, PETER; DONKOH, EMMANUEL (2022), “Anemia Detection using Palpable Palm Image Datasets from Ghana”, Mendeley Data, V1, doi: 10.17632/ccr8cm22vz.1
                    2. Asare, Justice Williams; APPIAHENE, PETER; DONKOH, EMMANUEL (2023), “CP-AnemiC (A Conjunctival Pallor) Dataset from Ghana”, Mendeley Data, V1, doi: 10.17632/m53vz6b7fx.1

## 🧠 Technology Stack
- Machine Learning Models:

    - Empirical Analysis Models: Trained using TensorFlow for anemia detection.
    - CycleGAN: Trained using PyTorch for synthetic data generation and augmentation.
- Backend: Flask (Python) for handling server requests and data processing.

- Frontend: HTML, CSS, and JavaScript for a responsive, user-friendly interface.

- Deployment: (Not Yet)



## 💻 How to Use

Follow these steps to set up and run HemoVeda on your local machine.

### 1. Clone the Repository
First, clone the HemoVeda repository to your local machine:
```bash
git clone https://github.com/YourUsername/HemoVeda.git
cd HemoVeda 
```
### 2. Set Up a Virtual Environment (Recommended)
Create and activate a virtual environment to manage dependencies:

```bash
python -m venv hemoveda-env
source hemoveda-env/bin/activate  # On Windows, use `hemoveda-env\Scripts\activate` 
```

### 3. Install Dependencies
Install the required packages listed in requirements.txt:

```bash
pip install -r requirements.txt 
```

### 4. Start the Flask App
Run the web application using Flask:

```bash
python app.py
```
## 🎉 Results

HemoVeda provides a risk assessment based on visual analysis and blood report data. The multi-input model ensures a robust and comprehensive assessment, helping individuals and clinicians take informed action against anemia.

## 🔍 Future Enhancements
- Live Image Capture and Prediction: Till now user is uploading image but we have plan to add the live capture. In this user may capture any image (e.g. Palm), the ROI will be automatically detected by a set of algorithms and then prediction.
- Expanded Dataset: Including more diverse images and clinical data for enhanced model training.
- Integration with Healthcare Portals: Enabling collaboration with healthcare providers and systems.


## 💬 Feedback
We are continually improving HemoVeda and would love your feedback. For suggestions, questions, or collaboration, reach out to us.