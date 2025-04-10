from flask import Flask, request, render_template
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline, BlipProcessor, BlipForConditionalGeneration
from scipy.special import softmax
from werkzeug.utils import secure_filename
from PIL import Image
import torch
import pytesseract
import whisper
import os

# Set Tesseract path
pytesseract.pytesseract.tesseract_cmd = r"C:/Program Files/Tesseract-OCR/tesseract.exe"

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# --- Model Loading with Debug Prints ---
print("Loading sentiment model...")
sentiment_model_name = "cardiffnlp/twitter-roberta-base-sentiment"
tokenizer = AutoTokenizer.from_pretrained(sentiment_model_name)
sentiment_model = AutoModelForSequenceClassification.from_pretrained(sentiment_model_name)
sentiment_labels = ['Negative', 'Neutral', 'Positive']
print("Sentiment model loaded.")

print("Loading Whisper model...")
whisper_model = whisper.load_model("medium")
print("Whisper model loaded.")

print("Loading BLIP model...")
blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
print("BLIP model loaded.")

print("Loading category classifier...")
category_classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
categories = ["Entertainment", "Politics", "Health", "Education", "Sports"]
print("Category classifier loaded.")

# --- Utility Functions ---
def predict_sentiment(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True)
    with torch.no_grad():
        outputs = sentiment_model(**inputs)
        scores = outputs.logits[0].numpy()
        probs = softmax(scores)
    max_index = probs.argmax()
    return sentiment_labels[max_index], float(probs[max_index])

def classify_comment(text):
    result = category_classifier(text, candidate_labels=categories)
    return result['labels'][0]

def transcribe_audio(audio_path):
    print(f"Transcribing audio: {audio_path}")
    result = whisper_model.transcribe(audio_path, task="translate")
    print(f"Transcription complete: {result['text']}")
    return result["text"]

def image_to_text(image_path):
    print(f"Processing image: {image_path}")
    image = Image.open(image_path).convert("RGB")
    extracted_text = pytesseract.image_to_string(image).strip()
    if extracted_text:
        print("Text found using OCR.")
        return extracted_text
    print("OCR failed. Trying BLIP for caption.")
    inputs = blip_processor(images=image, return_tensors="pt")
    with torch.no_grad():
        output = blip_model.generate(**inputs)
    caption = blip_processor.decode(output[0], skip_special_tokens=True)
    print(f"Caption generated: {caption}")
    return caption

# --- Flask Routes ---
@app.route('/', methods=['GET', 'POST'])
def index():
    sentiment = None
    confidence = None
    category = None
    comment = ""

    if request.method == 'POST':
        print("POST request received.")

        # Case 1: User typed a comment
        comment = request.form['comment'].strip()
        print(f"Typed comment: {comment}")

        # Case 2: Audio file
        audio_file = request.files.get('audio')
        if audio_file and audio_file.filename.endswith('.wav'):
            print(f"Audio file received: {audio_file.filename}")
            filename = secure_filename(audio_file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            audio_file.save(file_path)
            comment = transcribe_audio(file_path)

        # Case 3: Image file
        image_file = request.files.get('image')
        if image_file and image_file.filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            print(f"Image file received: {image_file.filename}")
            filename = secure_filename(image_file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            image_file.save(file_path)
            comment = image_to_text(file_path)

        if comment:
            print(f"Final comment to analyze: {comment}")
            sentiment, confidence = predict_sentiment(comment)
            print(f"Predicted sentiment: {sentiment} ({confidence:.2f})")
            category = classify_comment(comment)
            print(f"Predicted category: {category}")

    return render_template('index.html', sentiment=sentiment, confidence=confidence, comment=comment, category=category)

if __name__ == '__main__':
    print("Starting Flask app...")
    app.run(debug=True)
