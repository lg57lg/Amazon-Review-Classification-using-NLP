import pandas as pd
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# ===========================
# EXPANDED TRAINING DATA
# ===========================
data = {
    "Review": [
        # Positive Reviews (30)
        "This product is amazing",
        "Excellent quality and fast delivery",
        "Highly recommended for everyone",
        "Battery backup is outstanding",
        "The mobile phone works perfectly",
        "Very comfortable shoes",
        "The laptop performance is excellent",
        "Good packaging and original product",
        "Five stars for this amazing item",
        "Worth every penny",
        "Best purchase I've made",
        "Superb quality at great price",
        "Absolutely love this product",
        "Outstanding customer service",
        "Fast shipping and great condition",
        "Exactly as described",
        "Fantastic item, highly satisfied",
        "Premium quality, exceptional value",
        "Product exceeded my expectations",
        "Impressed with the quality",
        "Perfect fit and excellent craftsmanship",
        "Would buy again without hesitation",
        "Amazing value for money",
        "Professional quality and reliable",
        "Excellent performance and reliability",
        "Top notch product and service",
        "Very pleased with purchase",
        "Great quality at affordable price",
        "Brilliant product, works as advertised",
        "Cannot ask for better quality",
        
        # Negative Reviews (30)
        "Very poor quality",
        "Completely waste of money",
        "Battery drains very fast",
        "Received damaged product",
        "The charger stopped working",
        "Very disappointed with the purchase",
        "Bad customer support",
        "The product is fake",
        "Screen quality is terrible",
        "I will never buy this again",
        "Worst purchase ever",
        "Defective out of the box",
        "Cheap quality and poor materials",
        "Stopped working after one week",
        "Extremely disappointed",
        "Not worth the price",
        "Terrible durability",
        "Product broke immediately",
        "Waste of money and time",
        "False advertising",
        "Horrible build quality",
        "Faulty product received",
        "Regret buying this",
        "Poor quality materials",
        "Fell apart after few days",
        "Cannot recommend this product",
        "Terrible experience overall",
        "Disappointed with quality",
        "Does not work as described",
        "Complete disappointment"
    ],
    "Sentiment": [
        # Positive (30)
        "Positive", "Positive", "Positive", "Positive", "Positive",
        "Positive", "Positive", "Positive", "Positive", "Positive",
        "Positive", "Positive", "Positive", "Positive", "Positive",
        "Positive", "Positive", "Positive", "Positive", "Positive",
        "Positive", "Positive", "Positive", "Positive", "Positive",
        "Positive", "Positive", "Positive", "Positive", "Positive",
        
        # Negative (30)
        "Negative", "Negative", "Negative", "Negative", "Negative",
        "Negative", "Negative", "Negative", "Negative", "Negative",
        "Negative", "Negative", "Negative", "Negative", "Negative",
        "Negative", "Negative", "Negative", "Negative", "Negative",
        "Negative", "Negative", "Negative", "Negative", "Negative",
        "Negative", "Negative", "Negative", "Negative", "Negative"
    ]
}

# Create DataFrame
df = pd.DataFrame(data)

print(f"Total reviews: {len(df)}")
print(f"Positive reviews: {len(df[df['Sentiment'] == 'Positive'])}")
print(f"Negative reviews: {len(df[df['Sentiment'] == 'Negative'])}")
print("\n" + "="*50 + "\n")

# ===========================
# TRAIN-TEST SPLIT
# ===========================
X = df['Review']
y = df['Sentiment']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, 
    test_size=0.2, 
    random_state=42, 
    stratify=y
)

print(f"Training set size: {len(X_train)}")
print(f"Test set size: {len(X_test)}")
print(f"Training set - Positive: {(y_train == 'Positive').sum()}, Negative: {(y_train == 'Negative').sum()}")
print(f"Test set - Positive: {(y_test == 'Positive').sum()}, Negative: {(y_test == 'Negative').sum()}")
print("\n" + "="*50 + "\n")

# ===========================
# VECTORIZATION
# ===========================
vectorizer = TfidfVectorizer(
    max_features=100,
    min_df=1,
    max_df=0.9,
    lowercase=True,
    stop_words='english'
)

X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

print(f"Vocabulary size: {len(vectorizer.vocabulary_)}")
print(f"Features shape (train): {X_train_vec.shape}")
print("\n" + "="*50 + "\n")

# ===========================
# MODEL TRAINING
# ===========================
model = MultinomialNB(alpha=1.0)
model.fit(X_train_vec, y_train)

print(f"Model classes: {model.classes_}")
print(f"Class log priors: {model.class_log_prior_}")
print("\n" + "="*50 + "\n")

# ===========================
# EVALUATION
# ===========================
y_pred = model.predict(X_test_vec)
accuracy = accuracy_score(y_test, y_pred)

print(f"Test Accuracy: {accuracy * 100:.2f}%")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("\n" + "="*50 + "\n")

# ===========================
# SAVE MODEL & VECTORIZER
# ===========================
joblib.dump(vectorizer, "vectorizer.pkl")
joblib.dump(model, "model.pkl")

print("✅ Model and vectorizer saved successfully!")
print("\nModel classes (in order):", model.classes_)
print("This is important: the app will display predictions in this order.")

# ===========================
# TEST WITH EXAMPLES
# ===========================
print("\n" + "="*50)
print("TESTING WITH EXAMPLES")
print("="*50 + "\n")

test_reviews = [
    "This is an amazing product, excellent quality!",
    "Terrible product, waste of money",
    "good quality product i enjoy it",
    "Battery drains very fast, very disappointed"
]

for review in test_reviews:
    features = vectorizer.transform([review])
    pred = model.predict(features)[0]
    proba = model.predict_proba(features)[0]
    prob_dict = dict(zip(model.classes_, proba))
    
    print(f"Review: '{review}'")
    print(f"Prediction: {pred}")
    print(f"Confidence: {prob_dict[pred] * 100:.1f}%")
    print(f"All probabilities: {prob_dict}")
    print()
