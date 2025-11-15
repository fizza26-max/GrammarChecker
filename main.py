import streamlit as st
from PyPDF2 import PdfReader
from docx import Document
import pytesseract
from PIL import Image
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch
import re

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

# Function to split text into lines (sentences or paragraphs)
def split_into_lines(text):
    # Split by newlines or sentences (basic split)
    lines = re.split(r'(?<=[.!?])\s+', text.strip())
    return [line.strip() for line in lines if line.strip()]

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
    .highlight {
        background-color: #ffe6e6; /* Light red for highlighting errors */
        padding: 5px;
        border-radius: 3px;
        margin: 5px 0;
    }
    </style>
""", unsafe_allow_html=True)

# Streamlit UI
st.set_page_config(page_title="AI Grammar Checker", page_icon="📝", layout="centered")
st.title("AI Grammar Checker")

# Text input
text_input = st.text_area("Enter your text here:", height=200)

# File uploader
uploaded_file = st.file_uploader("Or upload a file:", type=["pdf", "docx", "jpeg", "jpg", "png"])

# Button to check grammar
if st.button("Check Grammar"):
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
        lines = split_into_lines(content_to_check)
        st.subheader("Grammar Check Results:")
        
        for i, line in enumerate(lines):
            corrected_line = fix_grammar(line)
            if corrected_line.strip().lower() != line.strip().lower():  # If there's a difference, mark as problematic
                st.markdown(f'<div class="highlight"><strong>Line {i+1} (Needs Correction):</strong> {line}</div>', unsafe_allow_html=True)
                if st.button(f"Correct Line {i+1}", key=f"correct_{i}"):
                    st.write(f"**Corrected:** {corrected_line}")
            else:
                st.write(f"**Line {i+1} (OK):** {line}")
    else:
        st.error("No text found to check. Please enter text or upload a valid file.")

# Footer
st.markdown("---")

