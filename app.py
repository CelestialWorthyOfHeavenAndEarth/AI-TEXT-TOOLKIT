import os
import requests
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

app = Flask(__name__)
CORS(app)

# Get API token from environment variables
API_TOKEN = os.environ.get("HUGGINGFACE_API_TOKEN")
if not API_TOKEN:
    print("Warning: HUGGINGFACE_API_TOKEN not found in environment variables")
    # Use a fallback for development only
    API_TOKEN = "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"  # Replace with env var in production

HEADERS = {"Authorization": f"Bearer {API_TOKEN}"}

MODELS = {
    "summarization": "facebook/bart-large-cnn",
    "sentiment": "distilbert-base-uncased-finetuned-sst-2-english",
    "keywords": "ml6team/keyphrase-extraction-distilbert-inspec",
    "classification": "facebook/bart-large-mnli",
    "qa": "distilbert-base-cased-distilled-squad",
    "generation": "gpt2"
}

def query_huggingface(model, payload, timeout=10):
    """Query Hugging Face API with error handling and timeout"""
    api_url = f"https://api-inference.huggingface.co/models/{model}"
    try:
        response = requests.post(api_url, headers=HEADERS, json=payload, timeout=timeout)
        response.raise_for_status()  # Raise an exception for HTTP errors
        return response.json(), None
    except requests.exceptions.Timeout:
        return None, "API request timed out. Please try again."
    except requests.exceptions.HTTPError as e:
        return None, f"HTTP Error: {e}"
    except requests.exceptions.RequestException as e:
        return None, f"Request Error: {e}"
    except ValueError:  # Includes JSONDecodeError
        return None, "Invalid response from API"

def validate_text(text):
    """Basic validation for text input"""
    if not text or not isinstance(text, str):
        return False, "Text must be a non-empty string"
    if len(text) > 10000:  # Set a reasonable limit
        return False, "Text exceeds maximum length of 10,000 characters"
    return True, None

@app.route('/')
def home():
    return render_template("index.html")

@app.route('/summarize', methods=['POST'])
def summarize():
    text = request.json.get("text", "")
    
    # Validate input
    valid, error_msg = validate_text(text)
    if not valid:
        return jsonify({"error": error_msg}), 400
        
    output, error = query_huggingface(MODELS["summarization"], {"inputs": text})
    
    if error:
        return jsonify({"error": error}), 500
        
    if isinstance(output, list) and output and "summary_text" in output[0]:
        return jsonify({"summary": output[0]["summary_text"]})
    else:
        return jsonify({"error": "Unexpected API response format"}), 500

@app.route('/sentiment', methods=['POST'])
def sentiment():
    text = request.json.get("text", "")
    
    valid, error_msg = validate_text(text)
    if not valid:
        return jsonify({"error": error_msg}), 400
        
    output, error = query_huggingface(MODELS["sentiment"], {"inputs": text})
    
    if error:
        return jsonify({"error": error}), 500
        
    if isinstance(output, list) and output and "label" in output[0]:
        score = output[0].get("score", 0)
        return jsonify({
            "sentiment": output[0]["label"],
            "confidence": f"{score:.2%}" if isinstance(score, float) else "Unknown"
        })
    else:
        return jsonify({"error": "Unexpected API response format"}), 500

@app.route('/keywords', methods=['POST'])
def keywords():
    text = request.json.get("text", "")
    
    valid, error_msg = validate_text(text)
    if not valid:
        return jsonify({"error": error_msg}), 400
        
    output, error = query_huggingface(MODELS["keywords"], {"inputs": text})
    
    if error:
        return jsonify({"error": error}), 500
        
    keywords = []
    if isinstance(output, list):
        keywords = [kw["word"] for kw in output if "word" in kw]
        
    return jsonify({"keywords": keywords})

@app.route('/classify', methods=['POST'])
def classify():
    text = request.json.get("text", "")
    
    valid, error_msg = validate_text(text)
    if not valid:
        return jsonify({"error": error_msg}), 400
        
    labels = ["Technology", "Health", "Education", "Sports", "Business", "Entertainment"]
    
    payload = {
        "inputs": text,
        "parameters": {"candidate_labels": labels}
    }
    
    output, error = query_huggingface(MODELS["classification"], payload)
    
    if error:
        return jsonify({"error": error}), 500
        
    if isinstance(output, dict) and "labels" in output and "scores" in output:
        # Return top 3 categories with scores
        results = []
        for i in range(min(3, len(output["labels"]))):
            results.append({
                "label": output["labels"][i],
                "score": f"{output['scores'][i]:.2%}"
            })
        return jsonify({"classifications": results})
    else:
        return jsonify({"error": "Unexpected API response format"}), 500

@app.route('/qa', methods=['POST'])
def qa():
    data = request.json
    question = data.get("question", "")
    context = data.get("context", "")
    
    # Validate inputs
    if not question:
        return jsonify({"error": "Question is required"}), 400
    
    valid, error_msg = validate_text(context)
    if not valid:
        return jsonify({"error": error_msg}), 400
    
    payload = {"inputs": {"question": question, "context": context}}
    output, error = query_huggingface(MODELS["qa"], payload)
    
    if error:
        return jsonify({"error": error}), 500
        
    if isinstance(output, dict) and "answer" in output:
        confidence = output.get("score", 0)
        return jsonify({
            "answer": output["answer"],
            "confidence": f"{confidence:.2%}" if isinstance(confidence, float) else "Unknown"
        })
    else:
        return jsonify({"error": "Unexpected API response format"}), 500

@app.route('/generate', methods=['POST'])
def generate():
    text = request.json.get("text", "")
    
    valid, error_msg = validate_text(text)
    if not valid:
        return jsonify({"error": error_msg}), 400
    
    # Add parameters to control generation
    payload = {
        "inputs": text,
        "parameters": {
            "max_length": 150,
            "num_return_sequences": 1,
            "temperature": 0.7
        }
    }
    
    output, error = query_huggingface(MODELS["generation"], payload)
    
    if error:
        return jsonify({"error": error}), 500
        
    if isinstance(output, list) and output and "generated_text" in output[0]:
        return jsonify({"generated": output[0]["generated_text"]})
    else:
        return jsonify({"error": "Unexpected API response format"}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')