import os
import gdown
from flask import Flask, request, jsonify
from flask_cors import CORS
from transformers import GPT2Tokenizer, GPT2LMHeadModel

app = Flask(__name__)
CORS(app, origins=['http://localhost:8080'])

# Replace 'YOUR_FILE_ID' with your actual file ID
MODEL_URL = 'https://drive.google.com/file/d/10rJ8GQ-_5eq2uGh5hEk5Tio7tZ6IbHpa/view?usp=sharing'
MODEL_PATH = './fine_tuned_model/model.safetensors'
TOKENIZER_NAME = 'gpt2'

def download_model():
    if not os.path.exists(MODEL_PATH):
        print("Downloading model...")
        os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
        gdown.download(MODEL_URL, MODEL_PATH, quiet=False)
        print("Model downloaded.")
    else:
        print("Model already exists.")

def load_model_and_tokenizer():
    global model, tokenizer
    download_model()
    print("Loading tokenizer...")
    tokenizer = GPT2Tokenizer.from_pretrained(TOKENIZER_NAME)
    print("Tokenizer loaded.")
    
    try:
        print("Loading model...")
        model = GPT2LMHeadModel.from_pretrained('./fine_tuned_model', from_tf=False, torch_dtype='auto', low_cpu_mem_usage=True)
        print("Model loaded successfully.")
    except Exception as e:
        print(f"Error loading model: {e}")

load_model_and_tokenizer()

@app.route('/')
def home():
    return "Hello, World!"

@app.route('/generate', methods=['POST'])
def generate():
    # Your code for generating response
    response = jsonify({'generated_text': 'Sample text'})
    return response

@app.route('/generate', methods=['OPTIONS'])
def options():
    # Respond to OPTIONS request (preflight)
    headers = {
        'Access-Control-Allow-Origin': 'http://localhost:8080',  # Replace with your frontend URL
        'Access-Control-Allow-Methods': 'POST',
        'Access-Control-Allow-Headers': 'Content-Type'
    }
    return '', 204, headers


if __name__ == '__main__':
    app.run(debug=True)