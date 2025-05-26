# app.py
from flask import Flask, render_template, request, redirect, url_for
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import os

app = Flask(__name__)
app.secret_key = os.urandom(24)

# Load model and tokenizer
model_path = "./fraud_detection_model"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path)

def predict_text(text):
    # Tokenize input
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=256,
        padding="max_length"
    )
    
    # Get prediction
    with torch.no_grad():
        outputs = model(**inputs)
    
    # Convert to probabilities
    probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
    confidence, prediction = torch.max(probs, dim=-1)
    
    return {
        "label": model.config.id2label[prediction.item()],
        "confidence": confidence.item()
    }

@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        text = request.form['text']
        if not text.strip():
            return render_template('index.html', error="Please enter some text")
            
        result = predict_text(text)
        return render_template('result.html', 
                             prediction=result['label'],
                             confidence=result['confidence'],
                             original_text=text)
    
    return render_template('index.html')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)