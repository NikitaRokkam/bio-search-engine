# ml_utils.py
import joblib
import os
import random

# --- Query Type Prediction ---
MODEL_PATH = os.path.join('ml_models', 'query_classifier_pipeline.joblib')
CLASSIFIER_MODEL = None

def load_classifier_model():
    """Loads the classifier model into memory."""
    global CLASSIFIER_MODEL
    if CLASSIFIER_MODEL is None:
        try:
            CLASSIFIER_MODEL = joblib.load(MODEL_PATH)
            print("Query classifier model loaded successfully.")
        except FileNotFoundError:
            print(f"Error: Classifier model not found at {MODEL_PATH}")
            print("Please run train_classifier.py first.")
            # In a real app, you might stop the server or handle this more gracefully
            CLASSIFIER_MODEL = None # Ensure it's None if loading fails
        except Exception as e:
            print(f"Error loading classifier model: {e}")
            CLASSIFIER_MODEL = None
    return CLASSIFIER_MODEL

def predict_query_type(query):
    """Predicts if a query is a 'sequence' or 'keyword'."""
    model = load_classifier_model()
    if model is None:
        # Fallback logic if model fails to load
        print("Classifier model not loaded. Using basic fallback rule.")
        # Very basic rule: check for spaces or significant number of digits
        if ' ' in query or sum(c.isdigit() for c in query) > 3:
            return "keyword"
        # Check character set (simple version)
        allowed_chars = set("ACGTUNWSMKRYBDHV") # DNA/RNA IUPAC
        if all(c.upper() in allowed_chars for c in query if c.isalnum()):
             return "sequence" # Likely DNA/RNA
        allowed_chars = set("ABCDEFGHIKLMNPQRSTUVWXYZ*") # Protein IUPAC + stop
        if all(c.upper() in allowed_chars for c in query if c.isalnum()):
            return "sequence" # Likely Protein
        return "keyword" # Default fallback

    # Use the loaded model
    try:
        # Model expects a list/iterable
        prediction = model.predict([query])
        return prediction[0] # Return the first (and only) prediction
    except Exception as e:
        print(f"Error during query type prediction: {e}")
        # Fallback if prediction fails
        return "keyword" # Or some other default


# --- Result Ranking (ML Part 2 - Placeholder/Heuristic) ---
# A real ML ranker needs training data (query, result, relevance_score)
# For simplicity, we'll use a heuristic approach first.
# ML approach is commented out as an advanced option.

def rank_results(results, query_type, query):
    """
    Ranks results based on heuristics.
    Prioritizes sources based on query type and potentially keyword matching.
    """
    print(f"Ranking {len(results)} results for query '{query}' (type: {query_type})")
    if not results:
        return []

    # Heuristic Ranking Rules:
    # 1. Give higher scores to sources typically better for the query type.
    # 2. Boost score if query keywords appear in the title/description.
    # 3. UniProt often good for proteins (sequence or keyword).
    # 4. GenBank good for genes/nucleotides (sequence or keyword).

    scored_results = []
    query_lower = query.lower()
    query_keywords = set(query_lower.split()) if query_type == 'keyword' else set()

    for result in results:
        score = 0
        source = result.get('source', 'Unknown').lower()
        title = result.get('title', '').lower()
        # description = result.get('description', '').lower() # Could add if available

        # Rule 1: Source preference based on query type
        if query_type == 'sequence':
            if 'uniprot' in source:
                score += 10 # Good for protein sequences
            elif 'genbank' in source or 'ncbi' in source:
                score += 8 # Good for nucleotide sequences
            else:
                score += 2 # Other sources
        elif query_type == 'keyword':
            if 'uniprot' in source:
                score += 8 # Good for protein keywords
            elif 'genbank' in source or 'ncbi' in source:
                score += 10 # Good for gene/general keywords
            else:
                score += 5 # Other sources

        # Rule 2: Keyword matching (only for keyword queries)
        if query_type == 'keyword' and query_keywords:
            title_words = set(title.split())
            common_words = query_keywords.intersection(title_words)
            if common_words:
                score += len(common_words) * 5 # Boost score based on keyword overlap

            # Bonus for exact phrase match (simple check)
            if query_lower in title:
                score += 15

        # Simple random tie-breaker or slight variance
        score += random.uniform(0, 0.1)

        scored_results.append({'score': score, 'data': result})

    # Sort results by score in descending order
    ranked_results = sorted(scored_results, key=lambda x: x['score'], reverse=True)

    print(f"Ranking scores sample: {[res['score'] for res in ranked_results[:5]]}")

    # Return just the original result dictionaries in the new order
    return [res['data'] for res in ranked_results]

"""
# --- Advanced ML Ranking (Conceptual - Requires Training Data) ---
RANKING_MODEL = None # Load a trained ranking model (e.g., RankSVM, LambdaMART)

def extract_ranking_features(result, query_type, query):
    # Extract features like:
    # - Source (one-hot encoded)
    # - Query type (binary)
    # - Keyword match score (e.g., TF-IDF similarity between query and title/desc)
    # - Result type (gene, protein, structure) if available
    # - Sequence length match (if query_type is sequence)
    # ... etc.
    features = [...]
    return features

def rank_results_ml(results, query_type, query):
    if not results or RANKING_MODEL is None:
        return rank_results_heuristic(results, query_type, query) # Fallback

    features_list = [extract_ranking_features(res, query_type, query) for res in results]
    # Predict relevance scores using the ML model
    scores = RANKING_MODEL.predict(features_list) # Or decision_function

    # Combine results with scores and sort
    scored_results = list(zip(scores, results))
    ranked_results = sorted(scored_results, key=lambda x: x[0], reverse=True)

    return [res for score, res in ranked_results]
"""