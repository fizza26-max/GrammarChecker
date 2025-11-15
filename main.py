import streamlit as st
from PyPDF2 import PdfReader
from docx import Document
import pytesseract
from PIL import Image
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch

# Load Hugging Face grammar correction model
@st.cache_resource(show_spinner=False)
def load_model():
    tokenizer = AutoTokenizer.from_pretrained("prithivida/grammar_error_correcter_v1")
    model = AutoModelForSeq2SeqLM.from_pretrained("prithivida/grammar_error_correcter_v1")
    return tokenizer, model

tokenizer, model = load_model()

# Function to correct grammar using the model
def fix_grammar(text):
    inputs = tokenizer.encode(text, return_tensors="pt", truncation=True, max_length=512)
    outputs = model.generate(inputs, max_length=512)
    corrected_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return corrected_text

# Function to extract text from PDF
def extract_text_from_pdf(file):
    reader = PdfReader(file)
    text = ""
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text + "\n"
    return text

# Function to extract text from DOCX
def extract_text_from_docx(file):
    doc = Document(file)
    text = "\n".join([para.text for para in doc.paragraphs])
    return text

# Function to extract text from image using OCR
def extract_text_from_image(file):
    image = Image.open(file)
    text = pytesseract.image_to_string(image)
    return text

# Custom CSS for blue and white theme
st.markdown("""
    <style>
    body {
        background-color: #ffffff;
        color: #003366;
    }
    .stButton>button {
        background-color: #0066cc;
        color: #ffffff;
        border: none;
        border-radius: 5px;
    }
    .stButton>button:hover {
        background-color: #004d99;
    }
    .stTextArea textarea {
        border: 1px solid #0066cc;
        border-radius: 5px;
        background-color: #ffffff;
        color: #003366;
    }
    .stFileUploader {
        border: 1px solid #0066cc;
        border-radius: 5px;
        background-color: #e6f2ff;
    }
    .stAlert {
        border-left: 5px solid #0066cc;
    }
    </style>
""", unsafe_allow_html=True)

# Streamlit UI
st.set_page_config(page_title="AI Grammar Checker", page_icon="📝", layout="centered")
st.title("AI Grammar Checker")
st.markdown("Enter text or upload a file (.pdf, .docx, .jpeg, .jpg, .png) for grammar correction.")

# Text input
text_input = st.text_area("Enter your text here:", height=200)

# File uploader
uploaded_file = st.file_uploader("Or upload a file:", type=["pdf", "docx", "jpeg", "jpg", "png"])

# Button to correct grammar
if st.button("Correct Grammar"):
    content_to_check = text_input.strip()

    if uploaded_file is not None:
        file_type = uploaded_file.type
        try:
            if file_type == "application/pdf":
                content_to_check = extract_text_from_pdf(uploaded_file)
            elif file_type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
                content_to_check = extract_text_from_docx(uploaded_file)
            elif file_type.startswith("image/"):
                content_to_check = extract_text_from_image(uploaded_file)
            else:
                st.error("Unsupported file type.")
                content_to_check = ""
        except Exception as e:
            st.error(f"Error processing file: {str(e)}")
            content_to_check = ""

    if content_to_check:
        with st.spinner("Correcting grammar..."):
            corrected_text = fix_grammar(content_to_check)

        st.subheader("Corrected Text:")
        st.text_area("", value=corrected_text, height=300)
    else:
        st.error("No text found to correct. Please enter text or upload a valid file.")

# Footer
st.markdown("---")
st.markdown("Powered by Hugging Face Transformers and Streamlit.")
