# train_model.py (Updated for 7 categories)
import os
import re
import joblib
import pandas as pd
import json
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from datetime import datetime

print("\n" + "=" * 60)
print("🚀 NEWS CLASSIFIER - MODEL TRAINING (7 CATEGORIES)")
print("=" * 60)

# --------------------- NLTK ---------------------
nltk.download("stopwords", quiet=True)
stop_words = set(stopwords.words("english"))
stemmer = PorterStemmer()


def preprocess(text):
    """Enhanced text preprocessing."""
    if not isinstance(text, str):
        return ""

    text = re.sub(r"http\S+|www\S+|https\S+", "", text)
    text = re.sub(r"[^a-zA-Z\s.!?]", " ", text)
    text = re.sub(r"\s+", " ", text)
    text = text.lower().split()
    text = [stemmer.stem(w) for w in text if w not in stop_words and len(w) > 2]
    return " ".join(text)


# --------------------- Load Dataset ---------------------
DATA_PATH = "News_Category_Dataset_v3.json"

print(f"\n📂 Loading dataset from: {DATA_PATH}")

try:
    # Try different JSON formats
    try:
        df = pd.read_json(DATA_PATH, lines=True)
        print("✅ Loaded as JSON lines format")
    except:
        df = pd.read_json(DATA_PATH)
        print("✅ Loaded as standard JSON format")

    print(f"📊 Dataset shape: {df.shape}")
    print(f"📋 Columns: {df.columns.tolist()}")

except Exception as e:
    print(f"❌ Error loading dataset: {e}")
    exit()

# --------------------- Data Preparation ---------------------
# Detect label column
label_cols = ["category", "Category", "CATEGORY"]
label_col = None
for col in label_cols:
    if col in df.columns:
        label_col = col
        break

if not label_col:
    print("❌ No category column found!")
    print(f"   Available columns: {df.columns.tolist()}")
    exit()

print(f"🎯 Using label column: '{label_col}'")

# Create text column
if "headline" in df.columns and "short_description" in df.columns:
    df["text"] = df["headline"] + ". " + df["short_description"]
    print("✅ Using 'headline + short_description' as text")
elif "text" in df.columns:
    print("✅ Using 'text' column")
elif "description" in df.columns:
    df["text"] = df["description"]
    print("✅ Using 'description' column")
else:
    # Use first text column
    text_cols = [c for c in df.columns if c != label_col and df[c].dtype == "object"]
    if text_cols:
        df["text"] = df[text_cols[0]]
        print(f"⚠️ Using '{text_cols[0]}' as text column")
    else:
        print("❌ No suitable text column found!")
        exit()

# Clean and prepare data
df = df[["text", label_col]].copy()
df = df.dropna()
df = df[df["text"].str.strip() != ""]

print(f"\n📊 Data after cleaning:")
print(f"   • Total samples: {len(df):,}")
print(f"   • Unique categories: {df[label_col].nunique()}")

# Show category distribution
print("\n📈 Category distribution (top 20):")
print(df[label_col].value_counts().head(20))

# --------------------- Preprocessing ---------------------
print("\n🔄 Preprocessing text...")
df["clean"] = df["text"].apply(preprocess)

X = df["clean"]
y = df[label_col]

# --------------------- Train-Test Split ---------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"\n🎯 Dataset split:")
print(f"   • Training set: {len(X_train):,} samples")
print(f"   • Test set: {len(X_test):,} samples")

# --------------------- Model Training ---------------------
print("\n🤖 Training model...")

model = Pipeline(
    [
        (
            "tfidf",
            TfidfVectorizer(
                max_features=10000, ngram_range=(1, 2), min_df=2, max_df=0.8
            ),
        ),
        ("svc", LinearSVC(C=1.0, max_iter=10000, random_state=42)),
    ]
)

model.fit(X_train, y_train)

# --------------------- Evaluation ---------------------
print("\n📊 Evaluating model...")
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print(f"\n✅ Model Performance:")
print(f"   • Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")

# Get classification report
print("\n🔹 Classification Report (top categories):")
report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
categories = list(report.keys())[:-3]  # Exclude avg/total

for cat in categories[:15]:  # Show top 15
    if cat in ["accuracy", "macro avg", "weighted avg"]:
        continue
    prec = report[cat]["precision"]
    rec = report[cat]["recall"]
    f1 = report[cat]["f1-score"]
    support = report[cat]["support"]
    print(
        f"   {cat[:20]:<20} | Precision: {prec:.3f} | Recall: {rec:.3f} | F1: {f1:.3f} | Samples: {support}"
    )

# --------------------- Save Model ---------------------
print("\n💾 Saving model and metadata...")
os.makedirs("models", exist_ok=True)

# Save model
MODEL_PATH = "models/news_classifier.pkl"
joblib.dump(model, MODEL_PATH)
print(f"   • Model saved to: {MODEL_PATH}")

# Save class labels
CLASSES_PATH = "models/class_labels.json"
with open(CLASSES_PATH, "w") as f:
    json.dump(model.classes_.tolist(), f)
print(f"   • Class labels saved to: {CLASSES_PATH}")

# Save label mapping for 7 categories
label_mapping = {
    # Politics
    "POLITICS": "Politics",
    "POLITICAL": "Politics",
    # Entertainment
    "ENTERTAINMENT": "Entertainment",
    "COMEDY": "Entertainment",
    "ARTS": "Entertainment",
    "ARTS & CULTURE": "Entertainment",
    # World News
    "WORLD NEWS": "World News",
    "WORLDPOST": "World News",
    "THE WORLDPOST": "World News",
    "U.S. NEWS": "World News",
    # Business
    "BUSINESS": "Business",
    "MONEY": "Business",
    # Science/Tech
    "SCIENCE": "Science/Tech",
    "TECH": "Science/Tech",
    "TECHNOLOGY": "Science/Tech",
    # Sports
    "SPORTS": "Sports",
    "SPORT": "Sports",
    # Others (common ones)
    "WELLNESS": "Others",
    "TRAVEL": "Others",
    "STYLE & BEAUTY": "Others",
    "PARENTING": "Others",
    "FOOD & DRINK": "Others",
    "HOME & LIVING": "Others",
}

MAPPING_PATH = "models/label_mapping.json"
with open(MAPPING_PATH, "w") as f:
    json.dump(label_mapping, f)
print(f"   • Label mapping saved to: {MAPPING_PATH}")

# Save training metadata
metadata = {
    "training_date": datetime.now().isoformat(),
    "dataset_size": len(df),
    "num_classes": len(model.classes_),
    "accuracy": float(accuracy),
    "target_categories": [
        "Politics",
        "Entertainment",
        "World News",
        "Business",
        "Science/Tech",
        "Sports",
        "Others",
    ],
    "model_type": "TF-IDF + LinearSVC",
    "features": "ngram_range=(1,2), max_features=10000",
}

METADATA_PATH = "models/training_metadata.json"
with open(METADATA_PATH, "w") as f:
    json.dump(metadata, f, indent=2)
print(f"   • Training metadata saved to: {METADATA_PATH}")

# --------------------- Test Predictions ---------------------
print("\n🧪 Testing with sample texts for 7 categories:")

test_samples = [
    # Politics
    "The President signed a new executive order on climate change today.",
    "Congress is debating the new immigration reform bill.",
    # Entertainment
    "Taylor Swift's new album broke streaming records worldwide.",
    "The Oscars ceremony featured stunning red carpet fashion.",
    # World News
    "UN Security Council holds emergency meeting on Ukraine crisis.",
    "Earthquake strikes Japan, tsunami warnings issued.",
    # Business
    "Stock market surges as tech companies report strong earnings.",
    "Federal Reserve announces interest rate hike to combat inflation.",
    # Science/Tech
    "NASA's rover discovers evidence of ancient water on Mars.",
    "New AI chatbot can pass medical licensing exams.",
    # Sports
    "Liverpool wins Champions League in dramatic penalty shootout.",
    "NBA finals set record for television viewership.",
    # Others (Wellness, Travel, etc.)
    "Meditation shown to reduce stress by 40% in new study.",
    "Italy named best travel destination for food lovers.",
]

print("\n" + "-" * 80)
category_counter = {
    "Politics": 0,
    "Entertainment": 0,
    "World News": 0,
    "Business": 0,
    "Science/Tech": 0,
    "Sports": 0,
    "Others": 0,
}

for sample in test_samples:
    cleaned = preprocess(sample)
    try:
        prediction = model.predict([cleaned])[0]
        # Map to 7 categories
        if "POLITIC" in prediction.upper():
            mapped = "Politics"
        elif (
            "ENTERTAIN" in prediction.upper()
            or "COMEDY" in prediction.upper()
            or "ARTS" in prediction.upper()
        ):
            mapped = "Entertainment"
        elif "WORLD" in prediction.upper() or "U.S." in prediction.upper():
            mapped = "World News"
        elif "BUSINESS" in prediction.upper() or "MONEY" in prediction.upper():
            mapped = "Business"
        elif "SCIENCE" in prediction.upper() or "TECH" in prediction.upper():
            mapped = "Science/Tech"
        elif "SPORT" in prediction.upper():
            mapped = "Sports"
        else:
            mapped = "Others"

        category_counter[mapped] += 1

        print(f"📝 '{sample[:50]}...'")
        print(f"   → Raw: {prediction}")
        print(f"   → 7-Category: {mapped}")
        print()

    except Exception as e:
        print(f"   ❌ Error predicting: {e}")

print("\n📊 Test Distribution:")
for cat, count in category_counter.items():
    print(f"   • {cat}: {count} samples")

print("\n" + "=" * 60)
print("🎯 TRAINING COMPLETE - 7 CATEGORIES READY!")
print("=" * 60)
print(f"   • Model saved in: models/")
print(f"   • Total fine-grained categories: {len(model.classes_)}")
print(f"   • Mapped to 7 categories: {', '.join(metadata['target_categories'])}")
print(f"   • Test accuracy: {accuracy*100:.2f}%")
print("=" * 60)
print("\n🚀 To run the application: python app.py")
print("🌐 Open http://localhost:5000 in your browser\n")
