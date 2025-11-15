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
    lines = re.split(r'(?<=[.!?])\s+', text.strip())
    return [line.strip() for line in lines if line.strip()]

# Custom CSS for blue and white theme and chat bubbles
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
    .user-message {
        background-color: #0066cc; /* Blue for user */
        color: #ffffff;
        padding: 10px;
        border-radius: 10px;
        margin: 5px 0;
        text-align: left;
        max-width: 70%;
        float: right;
        clear: both;
    }
    .ai-message {
        background-color: #ffffff; /* White for AI */
        color: #003366;
        border: 1px solid #0066cc;
        padding: 10px;
        border-radius: 10px;
        margin: 5px 0;
        text-align: left;
        max-width: 70%;
        float: left;
        clear: both;
    }
    .highlight {
        background-color: #ffe6e6; /* Light red for highlighting errors */
        padding: 5px;
        border-radius: 3px;
        margin: 5px 0;
    }
    .chat-container {
        overflow-y: auto;
        max-height: 400px;
        padding: 10px;
        border: 1px solid #0066cc;
        border-radius: 5px;
        background-color: #f9f9f9;
    }
    </style>
""", unsafe_allow_html=True)

# Sidebar for New Chat
with st.sidebar:
    st.title("Chat Options")
    if st.button("New Chat"):
        st.session_state.history = []
        st.rerun()

# Initialize session state for history
if 'history' not in st.session_state:
    st.session_state.history = []

# Streamlit UI
st.set_page_config(page_title="AI Grammar Checker Chatbot", page_icon="📝", layout="centered")
st.title("AI Grammar Checker Chatbot")
st.markdown("Chat with the AI to check and correct grammar in text or uploaded files. History is kept above.")

# Display chat history
st.subheader("Chat History")
chat_container = st.container()
with chat_container:
    for message in st.session_state.history:
        if message['role'] == 'user':
            st.markdown(f'<div class="user-message">{message["content"]}</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="ai-message">{message["content"]}</div>', unsafe_allow_html=True)

# Input area at the bottom
st.markdown("---")
col1, col2 = st.columns([3, 1])

with col1:
    text_input = st.text_area("Enter your text here:", height=100, key="input")

with col2:
    uploaded_file = st.file_uploader("Or upload a file:", type=["pdf", "docx", "jpeg", "jpg", "png"], key="file")

# Button to send message
if st.button("Send"):
    content_to_check = text_input.strip()
    user_message = f"Text: {text_input}" if text_input else ""

    if uploaded_file is not None:
        file_type = uploaded_file.type
        try:
            if file_type == "application/pdf":
                content_to_check = extract_text_from_pdf(uploaded_file)
                user_message += f" Uploaded PDF: {uploaded_file.name}"
            elif file_type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
                content_to_check = extract_text_from_docx(uploaded_file)
                user_message += f" Uploaded DOCX: {uploaded_file.name}"
            elif file_type.startswith("image/"):
                content_to_check = extract_text_from_image(uploaded_file)
                user_message += f" Uploaded Image: {uploaded_file.name}"
            else:
                user_message += " Unsupported file type."
                content_to_check = ""
        except Exception as e:
            user_message += f" Error processing file: {str(e)}"
            content_to_check = ""

    if content_to_check:
        # Add user message to history
        st.session_state.history.append({"role": "user", "content": user_message})

        # Process grammar check
        lines = split_into_lines(content_to_check)
        ai_response = "<strong>Grammar Check Results:</strong><br>"
        
        for i, line in enumerate(lines):
            corrected_line = fix_grammar(line)
            if corrected_line.strip().lower() != line.strip().lower():
                ai_response += f'<div class="highlight"><strong>Line {i+1} (Needs Correction):</strong> {line}</div>'
                # Note: Buttons in chat might not work perfectly; consider alternatives if needed
            else:
                ai_response += f"<strong>Line {i+1} (OK):</strong> {line}<br>"

        # Add AI response to history
        st.session_state.history.append({"role": "ai", "content": ai_response})

        # Clear inputs
        st.session_state.input = ""
        # Note: File uploader can't be cleared easily in Streamlit; user can re-upload if needed

        st.rerun()
    else:
        st.error("No text found. Please enter text or upload a valid file.")

# Footer
st.markdown("---")
st.markdown("Powered by Hugging Face Transformers and Streamlit.")
