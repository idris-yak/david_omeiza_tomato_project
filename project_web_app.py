# Importing the dependencies 
import streamlit as st
import tensorflow as tf
import numpy as np
import os
from PIL import Image
import io

# Set Streamlit page configuration
st.set_page_config(
    page_title="Tomato Disease Detection",
    layout="centered",
    initial_sidebar_state="expanded"
)

# Custom CSS to change background color
st.markdown("""
    <style>
    .main {
        background-color: #d8e4a1;
    }
    .css-1aumxhk, .css-1v3fvcr {
        background-color: #d8e4a1 !important;
    }
    </style>
""", unsafe_allow_html=True)

# Define class labels
class_names = [
    "bacterial spot", "early blight", "healthy tomato",
    "late blight", "southern blight"
]

@st.cache_resource
def load_model():
    model_path = "Project_Improved_Model2.keras"
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at: {model_path}")
    return tf.keras.models.load_model(model_path)

def tomato_disease_solution(disease):
    solutions = {
        "bacterial spot": """**Bacterial Spot Solution:**
- Use certified disease-free seeds.
- Avoid overhead watering.
- Apply copper-based bactericides.
- Remove infected plant debris.
- Ensure good air circulation.""",

        "early blight": """**Early Blight Solution:**
- Rotate crops.
- Use resistant varieties.
- Remove infected leaves.
- Apply fungicides (chlorothalonil/copper-based).
- Promote air flow between plants.""",

        "healthy tomato": """**Healthy Tomato Maintenance:**
- Water at the base, not on leaves.
- Use mulch to prevent soil splash.
- Regular balanced fertilization.
- Prune regularly.
- Monitor frequently for early signs of issues.""",

        "late blight": """**Late Blight Solution:**
- Use resistant cultivars.
- Destroy infected plants immediately.
- Apply mancozeb/chlorothalonil fungicides.
- Avoid overhead irrigation.
- Practice crop rotation.""",

        "southern blight": """**Southern Blight Solution:**
- Rotate crops and avoid soilborne buildup.
- Apply fungicides like PCNB.
- Remove infected debris.
- Deep plowing to bury sclerotia.
- Maintain well-drained soil."""
    }
    return solutions.get(disease, "No solution available for the detected condition.")

def predict(model, img):
    try:
        img = img.resize((256, 256))
        img_array = tf.keras.preprocessing.image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0) / 255.0

        predictions = model.predict(img_array)
        index = np.argmax(predictions[0])
        label = class_names[index]
        confidence = round(100 * float(predictions[0][index]), 2)
        solution = tomato_disease_solution(label)
        return label, confidence, solution
    except Exception as e:
        st.error(f"Error in prediction: {e}")
        return None, None, None

# Load model
model = load_model()

# Sidebar UI
st.sidebar.image("Logo.jpg", use_column_width=True)
st.sidebar.title("Dashboard")
page = st.sidebar.selectbox("Navigation", ["Home", "Prediction", "About", "FAQ"])

# HOME PAGE
if page == "Home":
    st.image("toma.jpg", width=200, caption="Tomato Disease Classifier", use_column_width=True)
    st.header("Welcome to the Tomato Disease Classification System 🍅")
    st.markdown("""
    Use this tool to detect and manage tomato plant diseases using AI-powered image analysis.
    - Upload or capture an image of a tomato leaf.
    - Get instant disease prediction and treatment tips.
    """)

# PREDICTION PAGE
elif page == "Prediction":
    st.subheader("Disease Detection 🔍")
    st.markdown("Upload a tomato leaf image or capture using your camera:")

    col1, col2 = st.columns(2)
    camera_file = col1.camera_input("Capture Image")
    uploaded_file = col2.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])

    img = None
    if camera_file:
        img = Image.open(io.BytesIO(camera_file.read()))
    elif uploaded_file:
        img = Image.open(uploaded_file)

    if img:
        st.image(img, caption="Image Preview", use_column_width=True)
        if st.button("Predict"):
            with st.spinner("Analyzing..."):
                label, confidence, solution = predict(model, img)
                if label:
                    st.success(f"Prediction: **{label}** ({confidence}%)")
                    st.info(solution)
    else:
        st.warning("Please upload or capture an image to proceed.")

# ABOUT PAGE
elif page == "About":
    st.subheader("About the Project")
    st.markdown("""
    This application is built to assist farmers and researchers in early detection of tomato leaf diseases. It utilizes a deep learning model trained on labeled images of various tomato diseases. 
    """)
    st.markdown("**Model Type**: Convolutional Neural Network (CNN)\n\n**Framework**: TensorFlow/Keras\n\n**Interface**: Streamlit")

    st.markdown("**Team Contacts:**")
    st.markdown("""
    - **Email**: support@davischoice@gmail.com  
    - **Phone**: +234-7064206404  
    - **Location**: Lagos State, Nigeria
    """)

# FAQ PAGE
elif page == "FAQ":
    st.subheader("Frequently Asked Questions")

    faq_list = [
        ("How does the system work?", "It uses a trained AI model to classify tomato leaf diseases based on images."),
        ("What diseases are covered?", "Currently: bacterial spot, early blight, late blight, southern blight, and healthy leaves."),
        ("Can I trust the prediction?", "While highly accurate, always cross-verify with a local agricultural expert."),
        ("Is my data stored?", "No. Uploaded images are processed in memory and not stored."),
        ("What if I encounter an error?", "Check your image quality or contact support if the issue persists."),
    ]

    for q, a in faq_list:
        with st.expander(q):
            st.write(a)
