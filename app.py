# app.py (Updated for 7 categories)
import os
import re
import joblib
import json
import nltk
from flask import Flask, render_template, request, jsonify
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import sent_tokenize

app = Flask(__name__)

# ------------------- NLTK SETUP --------------------
nltk.download("stopwords", quiet=True)
nltk.download("punkt", quiet=True)
english_stopwords = set(stopwords.words("english"))
stemmer = PorterStemmer()


# ------------------- MODEL LOADING ---`-----------------
def load_model_and_config():
    """Load model, classes, and configuration"""
    MODEL_PATH = os.path.join("models", "news_classifier.pkl")
    MAPPING_PATH = os.path.join("models", "label_mapping.json")

    if not os.path.exists(MODEL_PATH):
        print(f"❌ Model not found at {MODEL_PATH}")
        print("   Please run train_model.py first to create the model")
        return None, [], {}

    try:
        # Load model
        model = joblib.load(MODEL_PATH)
        print(f"✅ Model loaded successfully")

        # Get classes directly from model
        model_classes = list(model.classes_)
        print(f"   Found {len(model_classes)} classes in model")
        print(f"   Sample classes: {model_classes[:10]}")

        # Try to load label mapping if exists
        label_mapping = {}
        if os.path.exists(MAPPING_PATH):
            with open(MAPPING_PATH, "r") as f:
                label_mapping = json.load(f)
            print(f"   Label mapping loaded with {len(label_mapping)} entries")
        else:
            print("   No label mapping found, using default mapping")

        return model, model_classes, label_mapping

    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return None, [], {}


# Load model on startup
try:
    model, MODEL_CLASSES, LABEL_MAPPING = load_model_and_config()
    if model is None:
        print("⚠️ Model not loaded. The application will not work properly.")
        print("   Please run: python train_model.py")
except Exception as e:
    print(f"❌ Fatal error during model loading: {e}")
    model = None
    MODEL_CLASSES = []
    LABEL_MAPPING = {}


# ------------------- TEXT PROCESSING --------------------
def preprocess_text(text: str) -> str:
    """Clean + lowercase + remove stopwords + stemming."""
    if not isinstance(text, str):
        return ""

    # Remove URLs and special characters
    text = re.sub(r"http\S+|www\S+|https\S+", "", text)
    text = re.sub(r"[^a-zA-Z\s.!?]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()

    if not text:
        return ""

    text = text.lower().split()
    filtered = [
        stemmer.stem(t) for t in text if t not in english_stopwords and len(t) > 2
    ]
    return " ".join(filtered)


def split_sentences(text: str):
    """Improved sentence splitting using NLTK's sent_tokenize."""
    text = str(text).strip()
    if not text:
        return []

    # Use NLTK's sentence tokenizer for better accuracy
    sentences = sent_tokenize(text)
    return [s.strip() for s in sentences if s.strip()]


# ------------------- UPDATED TOPIC MAPPING (7 CATEGORIES) --------------------
# Now we have 7 clear categories:
# 1. Politics
# 2. Entertainment
# 3. World News
# 4. Business
# 5. Science/Tech
# 6. Sports
# 7. Others

DEFAULT_MAPPING = {
    # POLITICS
    "POLITICS": "Politics",
    "POLITICAL": "Politics",
    "GOVERNMENT": "Politics",
    "ELECTION": "Politics",
    "CONGRESS": "Politics",
    # ENTERTAINMENT
    "ENTERTAINMENT": "Entertainment",
    "COMEDY": "Entertainment",
    "ARTS": "Entertainment",
    "ARTS & CULTURE": "Entertainment",
    "CULTURE": "Entertainment",
    "MOVIES": "Entertainment",
    "TELEVISION": "Entertainment",
    "CELEBRITY": "Entertainment",
    "MUSIC": "Entertainment",
    # WORLD NEWS
    "WORLD NEWS": "World News",
    "WORLDPOST": "World News",
    "THE WORLDPOST": "World News",
    "U.S. NEWS": "World News",
    "INTERNATIONAL": "World News",
    "GLOBAL": "World News",
    "FOREIGN": "World News",
    # BUSINESS
    "BUSINESS": "Business",
    "MONEY": "Business",
    "ECONOMY": "Business",
    "FINANCE": "Business",
    "STOCK": "Business",
    "MARKET": "Business",
    "INVESTMENT": "Business",
    "COMPANY": "Business",
    "CORPORATE": "Business",
    # SCIENCE/TECH
    "SCIENCE": "Science/Tech",
    "TECH": "Science/Tech",
    "TECHNOLOGY": "Science/Tech",
    "SCI/TECH": "Science/Tech",
    "INNOVATION": "Science/Tech",
    "RESEARCH": "Science/Tech",
    "DIGITAL": "Science/Tech",
    "AI": "Science/Tech",
    "ARTIFICIAL INTELLIGENCE": "Science/Tech",
    # SPORTS
    "SPORTS": "Sports",
    "SPORT": "Sports",
    "FOOTBALL": "Sports",
    "BASKETBALL": "Sports",
    "BASEBALL": "Sports",
    "TENNIS": "Sports",
    "OLYMPICS": "Sports",
    "ATHLETE": "Sports",
    "GAME": "Sports",
    "CHAMPIONSHIP": "Sports",
    # OTHERS (will catch everything else)
    "WELLNESS": "Others",
    "HEALTH": "Others",
    "TRAVEL": "Others",
    "STYLE": "Others",
    "BEAUTY": "Others",
    "FASHION": "Others",
    "PARENTING": "Others",
    "FOOD": "Others",
    "DRINK": "Others",
    "RECIPE": "Others",
    "HOME": "Others",
    "LIVING": "Others",
    "EDUCATION": "Others",
    "SCHOOL": "Others",
    "RELIGION": "Others",
    "ENVIRONMENT": "Others",
    "GREEN": "Others",
    "CLIMATE": "Others",
    "IMPACT": "Others",
    "SOCIAL": "Others",
    "CRIME": "Others",
    "LAW": "Others",
    "WEDDINGS": "Others",
    "DIVORCE": "Others",
    "RELATIONSHIPS": "Others",
}


def map_to_main_topic(label: str) -> str:
    """
    Map dataset label -> one of 7 categories:
      Politics, Entertainment, World News, Business, Science/Tech, Sports, Others
    """
    if not label:
        return "Others"

    label_upper = str(label).upper()

    # First check the loaded mapping
    if label_upper in LABEL_MAPPING:
        return LABEL_MAPPING[label_upper]

    # Check default mapping (exact matches)
    if label_upper in DEFAULT_MAPPING:
        return DEFAULT_MAPPING[label_upper]

    # Check for partial matches in keywords
    # Politics keywords
    politics_keywords = [
        "POLITIC",
        "GOVERN",
        "ELECT",
        "SENATE",
        "PARLIAMENT",
        "MINISTER",
        "PRESIDENT",
        "CONGRESS",
    ]
    if any(keyword in label_upper for keyword in politics_keywords):
        return "Politics"

    # Entertainment keywords
    entertainment_keywords = [
        "ENTERTAIN",
        "MOVIE",
        "FILM",
        "TV",
        "CELEB",
        "ACTOR",
        "ACTRESS",
        "MUSIC",
        "SHOW",
        "THEATER",
    ]
    if any(keyword in label_upper for keyword in entertainment_keywords):
        return "Entertainment"

    # World News keywords
    world_keywords = [
        "WORLD",
        "INTERNATIONAL",
        "GLOBAL",
        "FOREIGN",
        "NATION",
        "COUNTRY",
    ]
    if any(keyword in label_upper for keyword in world_keywords):
        return "World News"

    # Business keywords
    business_keywords = [
        "BUSINESS",
        "ECONOM",
        "FINANC",
        "STOCK",
        "MARKET",
        "TRADE",
        "INDUSTRY",
        "COMPANY",
        "CORPORATION",
    ]
    if any(keyword in label_upper for keyword in business_keywords):
        return "Business"

    # Science/Tech keywords
    tech_keywords = [
        "SCIENCE",
        "TECH",
        "DIGITAL",
        "COMPUTER",
        "SOFTWARE",
        "INTERNET",
        "MOBILE",
        "ROBOT",
        "AI",
        "DATA",
    ]
    if any(keyword in label_upper for keyword in tech_keywords):
        return "Science/Tech"

    # Sports keywords
    sports_keywords = [
        "SPORT",
        "FOOTBALL",
        "BASKETBALL",
        "BASEBALL",
        "TENNIS",
        "GOLF",
        "ATHLET",
        "COACH",
        "TEAM",
        "PLAYER",
    ]
    if any(keyword in label_upper for keyword in sports_keywords):
        return "Sports"

    return "Others"


CATEGORIES = [
    "Politics",
    "Entertainment",
    "World News",
    "Business",
    "Science/Tech",
    "Sports",
    "Others",
]


# ------------------- FLASK ROUTES --------------------
@app.route("/", methods=["GET", "POST"])
def index():
    grouped = {cat: [] for cat in CATEGORIES}
    error_message = None
    total_sentences = 0
    model_status = "Loaded" if model else "Not Loaded"

    # Debug: Show what categories the model knows
    if model and MODEL_CLASSES:
        print(f"\n🔍 Model knows these {len(MODEL_CLASSES)} categories:")
        print("   " + ", ".join(MODEL_CLASSES[:15]) + "...")

    if request.method == "POST":
        if model is None:
            error_message = """
            ❌ Model not loaded. Please train the model first.<br>
            Run this command in your terminal: <code>python train_model.py</code>
            """
        else:
            text_input = request.form.get("news_text", "").strip()
            file = request.files.get("file")

            sentences = []

            # Process textarea input
            if text_input:
                sentences.extend(split_sentences(text_input))

            # Process file input
            if file and file.filename:
                if file.filename.endswith(".txt"):
                    try:
                        content = file.read().decode("utf-8", errors="ignore")
                        sentences.extend(split_sentences(content))
                    except Exception as e:
                        error_message = f"Error reading file: {str(e)}"
                else:
                    error_message = "Please upload only .txt files"

            # Filter and process sentences
            sentences = [s for s in sentences if s.strip()]
            total_sentences = len(sentences)

            if sentences and not error_message:
                try:
                    cleaned = [preprocess_text(s) for s in sentences]
                    preds = model.predict(cleaned)

                    print(f"\n🔍 Processing {total_sentences} sentences:")
                    print("-" * 60)

                    for sent, raw_label in zip(sentences, preds):
                        main_topic = map_to_main_topic(raw_label)
                        grouped[main_topic].append(sent)

                        # Debug output
                        print(f"📝 '{sent[:60]}...'")
                        print(f"   → Raw prediction: {raw_label}")
                        print(f"   → Mapped to: {main_topic}")
                        print()

                    print(f"✅ Classification complete!")
                    print(
                        f"📊 Distribution: "
                        + ", ".join(
                            [
                                f"{cat}: {len(grouped[cat])}"
                                for cat in CATEGORIES
                                if grouped[cat]
                            ]
                        )
                    )

                except Exception as e:
                    error_message = f"Prediction error: {str(e)}"
                    print(f"❌ Prediction error: {e}")

    return render_template(
        "dashboard.html",
        grouped=grouped,
        categories=CATEGORIES,
        error_message=error_message,
        total_sentences=total_sentences,
        model_status=model_status,
    )


@app.route("/train", methods=["GET"])
def train_model():
    """Route to train the model from the web interface"""
    try:
        import subprocess

        result = subprocess.run(
            ["python", "train_model.py"], capture_output=True, text=True
        )

        if result.returncode == 0:
            # Reload model after training
            global model, MODEL_CLASSES, LABEL_MAPPING
            model, MODEL_CLASSES, LABEL_MAPPING = load_model_and_config()
            return jsonify(
                {
                    "success": True,
                    "message": "Model trained successfully!",
                    "output": result.stdout,
                }
            )
        else:
            return jsonify(
                {"success": False, "message": "Training failed", "error": result.stderr}
            )
    except Exception as e:
        return jsonify({"success": False, "message": f"Error: {str(e)}"})


@app.route("/model-info", methods=["GET"])
def model_info():
    """Get information about the loaded model"""
    info = {
        "loaded": model is not None,
        "num_classes": len(MODEL_CLASSES) if model else 0,
        "categories": CATEGORIES,
        "sample_classes": MODEL_CLASSES[:10] if MODEL_CLASSES else [],
    }
    return jsonify(info)


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("📰 NEWS CLASSIFIER - 7 CATEGORIES")
    print("=" * 60)
    print(
        "Categories: Politics, Entertainment, World News, Business, Science/Tech, Sports, Others"
    )
    print("-" * 60)

    if model:
        print(f"✅ Model Status: LOADED")
        print(f"   Number of classes: {len(MODEL_CLASSES)}")
        print(f"   Target categories: {', '.join(CATEGORIES)}")
    else:
        print("❌ Model Status: NOT LOADED")
        print("   To train the model, run: python train_model.py")
        print("   Make sure 'News_Category_Dataset_v3.json' is in the project folder")

    print("\n🌐 Starting web server...")
    print("   Open http://localhost:5000 in your browser")
    print("=" * 60)

    app.run(host="0.0.0.0", port=5000, debug=True)
