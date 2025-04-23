# train_classifier.py
import pandas as pd
import re
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report
import joblib
import os

# --- 1. Sample Data ---
# In a real scenario, you'd want a much larger and more diverse dataset
data = {
    'query': [
        "ATGCGTGCATGCGTACGTAGCTAGCTAGCTAG", # DNA
        "MVHLTPEEKSAVTALWGKVNVDEVGGEALGRLLVVYPWTQRFFASFGNLSSPTAILGNPMVRAHGKKVLTSFGDAVKNLDNIKNTFSQLSELHCDKLHVDPENFRLLGNVLVCVLARNFGKEFTPQMQAAYQKVVAGVANALAHKYH", # Protein
        "cancer development", # Keyword
        "insulin receptor", # Keyword
        "GATTACA", # DNA (short)
        "MRVLKFGGTSVANAERFLRVADILESNARQGQVATVLSAPAKITNHLVAMIEKTISGQDALPNISDAERIFAELLTGLAAAQPGFPLAQLKTFVDQEFAQIKHVLHGISLLGQCPDSINAALICRGEKMSIAIMAGVLEARGHNVTVIDPVEKLLAVGHYLLE", # Protein
        "photosynthesis", # Keyword
        "signal transduction pathway", # Keyword
        "AAAAAAA", # DNA (simple repeat)
        "WWWWWWW", # Protein (simple repeat)
        "response to stimulus", # Keyword
        "ATGGCCAAGGAG", # DNA
        "P05067", # UniProt ID (treat as keyword initially)
        "NM_000059.4", # GenBank ID (treat as keyword initially)
        "BRCA1 gene", # Keyword
        "YLLGDPVT", # Short Peptide (could be seq or keyword fragment?) - tricky! Treat as seq for now.
        "1A2B", # PDB ID (treat as keyword initially)
        "structure AND protein", # Keyword phrase
    ],
    'type': [
        "sequence", "sequence", "keyword", "keyword", "sequence",
        "sequence", "keyword", "keyword", "sequence", "sequence",
        "keyword", "sequence", "keyword", "keyword", "keyword",
        "sequence", "keyword", "keyword"
    ]
}
df = pd.DataFrame(data)

# --- 2. Feature Engineering Function ---
# Simple features: length, character composition
def extract_features(query):
    query = query.upper()
    length = len(query)
    # Allowed DNA/RNA chars: A, C, G, T, U, N
    dna_rna_chars = set("ACGTUN")
    # Allowed Protein chars: A, C, D, E, F, G, H, I, K, L, M, N, P, Q, R, S, T, V, W, Y
    protein_chars = set("ACDEFGHIKLMNPQRSTVWY")
    # Generic biological chars (including ambiguous)
    bio_chars = dna_rna_chars.union(protein_chars).union(set("X*-."))

    bio_char_count = sum(1 for char in query if char in bio_chars)
    non_bio_char_count = length - bio_char_count
    digit_count = sum(1 for char in query if char.isdigit())
    space_count = sum(1 for char in query if char.isspace())

    # Calculate percentages
    percent_bio = (bio_char_count / length) * 100 if length > 0 else 0
    percent_non_bio = (non_bio_char_count / length) * 100 if length > 0 else 0
    percent_digits = (digit_count / length) * 100 if length > 0 else 0
    percent_spaces = (space_count / length) * 100 if length > 0 else 0

    # Simple rule: If high % of bio chars and low spaces/digits, likely sequence.
    # This logic is simple; a model learns more complex patterns.
    # We provide these features *to* the model.

    # For TF-IDF based model, we just return the text,
    # but let's keep these features in mind if TF-IDF isn't enough.
    # For this example, we'll use TF-IDF on the raw query string.
    # A more advanced approach might combine TF-IDF with these engineered features.
    return query # Return the preprocessed query for TF-IDF


# --- 3. Model Training ---
X = df['query']
y = df['type']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Create a pipeline: TF-IDF Vectorizer -> Naive Bayes Classifier
# TF-IDF is good for text classification, even works okay here.
# It looks at character n-grams (analyzer='char', ngram_range=(1, 3))
# which helps capture sequence-like patterns vs word patterns.
model_pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(analyzer='char', ngram_range=(1, 4))), # Use char n-grams
    ('clf', MultinomialNB()), # Naive Bayes is simple and often works well for text
])

print("Training model...")
model_pipeline.fit(X_train, y_train)
print("Training complete.")

# --- 4. Evaluate ---
print("\nEvaluating model...")
y_pred = model_pipeline.predict(X_test)
print(classification_report(y_test, y_pred))

# --- 5. Save the Model ---
model_dir = 'ml_models'
if not os.path.exists(model_dir):
    os.makedirs(model_dir)

model_path = os.path.join(model_dir, 'query_classifier_pipeline.joblib')
joblib.dump(model_pipeline, model_path)
print(f"\nModel saved to {model_path}")

# --- Test the saved model ---
loaded_model = joblib.load(model_path)
print("\nTesting loaded model with examples:")
test_queries = ["ACGTACGT", "insulin gene", "P53_HUMAN", "LLVVYPWTQRFFASFGNL"]
predictions = loaded_model.predict(test_queries)
for query, pred in zip(test_queries, predictions):
    print(f"Query: '{query}' -> Predicted Type: '{pred}'")
    