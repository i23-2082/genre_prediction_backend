from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import pickle
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import logging
from googletrans import Translator
from gtts import gTTS
import os
import uuid
import time
from threading import Thread
import requests

# Download NLTK data
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)

# Logging setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Supported languages for TTS (gTTS)
SUPPORTED_LANGUAGES = [
    'af', 'ar', 'bn', 'bs', 'ca', 'cs', 'cy', 'da', 'de', 'el', 'en', 'eo', 'es', 'et', 'fi',
    'fr', 'gu', 'hi', 'hr', 'hu', 'id', 'is', 'it', 'ja', 'jw', 'km', 'kn', 'ko', 'la', 'lv',
    'ml', 'mr', 'my', 'ne', 'nl', 'no', 'pl', 'pt', 'ro', 'ru', 'si', 'sk', 'sq', 'sr', 'su',
    'sv', 'sw', 'ta', 'te', 'th', 'tl', 'tr', 'uk', 'ur', 'vi', 'zh-CN', 'zh-TW'
]

# Load models with error handling
genre_classifier = None
vectorizer = None
mlb = None
try:
    with open('models/genre_classifier.pkl', 'rb') as f:
        genre_classifier = pickle.load(f)
    with open('models/vectorizer.pkl', 'rb') as f:
        vectorizer = pickle.load(f)
    with open('models/mlb.pkl', 'rb') as f:
        mlb = pickle.load(f)
    logger.info("Models loaded successfully")
except FileNotFoundError as e:
    logger.error(f"Model file not found: {e}")
    raise
except Exception as e:
    logger.error(f"Failed to load models: {e}")
    raise

# Init app and services
app = Flask(__name__)
CORS(app, resources={r"/api/*": {"origins": "*"}})
translator = Translator()
stop_words = set(stopwords.words('english'))

# Load default input from input.txt if it exists
def load_default_input():
    default_input = ""
    try:
        if os.path.exists('input.txt'):
            with open('input.txt', 'r', encoding='utf-8') as f:
                default_input = f.read().strip()
            logger.info("Loaded default input from input.txt")
    except Exception as e:
        logger.error(f"Failed to load input.txt: {e}")
    return default_input

# ----------- UTILS ------------
def preprocess_text(text):
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    text = text.lower()
    words = word_tokenize(text)
    words = [word for word in words if word not in stop_words]
    return ' '.join(words)

def predict_genres(summary, threshold=0.3):
    if not genre_classifier or not vectorizer or not mlb:
        raise ValueError("Models not loaded properly")
    processed = preprocess_text(summary)
    features = vectorizer.transform([processed])
    probabilities = genre_classifier.predict_proba(features)
    genres = [mlb.classes_[i] for i, prob in enumerate(probabilities[0]) if prob >= threshold]
    return genres if genres else ['None']

# ----------- ROUTES ------------
@app.route('/api/hello', methods=['GET'])
def hello():
    return jsonify({'message': 'Hello, World!'}), 200

@app.route('/api/predict', methods=['POST'])
def predict():
    data = request.get_json()
    if not data or 'input' not in data:
        default_input = load_default_input()
        if not default_input:
            return jsonify({'error': 'Missing input data'}), 400
        summary = default_input
    else:
        summary = data['input'].strip()
        if not summary:
            default_input = load_default_input()
            summary = default_input if default_input else ""

    if not summary:
        return jsonify({'error': 'Input cannot be empty'}), 400

    try:
        genres = predict_genres(summary)
        return jsonify({'genres': genres}), 200
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        return jsonify({'error': f'Prediction failed: {str(e)}'}), 500

@app.route('/api/tts', methods=['POST'])
def tts():
    data = request.get_json()
    text = data.get('text', '').strip()
    lang = data.get('language', 'en').strip()
    translate_only = data.get('translate_only', False)

    if not text:
        return jsonify({'error': 'No text provided'}), 400

    if lang not in SUPPORTED_LANGUAGES:
        return jsonify({'error': f'Unsupported language code: {lang}'}), 400

    filename = None
    try:
        translated = translator.translate(text, dest=lang)
        if not translated or not translated.text:
            raise ValueError("Translation returned empty result")
        translated_text = translated.text
        logger.info(f"Translated to {lang}: {translated_text}")

        if translate_only:
            return jsonify({'translated_text': translated_text}), 200

        # Use a temporary directory for Vercel's ephemeral filesystem
        temp_dir = '/tmp' if os.path.exists('/tmp') else '.'
        filename = os.path.join(temp_dir, f"audio_{uuid.uuid4()}.mp3")
        tts = gTTS(text=translated_text, lang=lang, slow=False)
        tts.save(filename)

        time.sleep(0.5)
        if not os.path.exists(filename):
            raise FileNotFoundError("Audio file was not saved")

        return send_file(filename, mimetype='audio/mpeg', as_attachment=False)

    except requests.exceptions.RequestException as e:
        logger.error(f"Translation network error: {e}")
        return jsonify({'error': 'Translation failed due to network issues'}), 500
    except ValueError as e:
        logger.error(f"Translation error: {e}")
        return jsonify({'error': str(e)}), 500
    except Exception as e:
        logger.error(f"TTS error: {e}")
        return jsonify({'error': f'Operation failed: {str(e)}'}), 500
    finally:
        if not translate_only and filename and os.path.exists(filename):
            def delete_file(path):
                try:
                    time.sleep(2)
                    if os.path.exists(path):
                        os.remove(path)
                        logger.info(f"Deleted file: {path}")
                except Exception as e:
                    logger.error(f"File delete error: {str(e)}")
            Thread(target=delete_file, args=(filename,)).start()
