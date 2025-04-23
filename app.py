# app.py
from flask import Flask, render_template, request, redirect, url_for
import ml_utils   # Our ML functions
import fetchers   # Our data fetching functions

app = Flask(__name__)

# Load the ML model when the app starts
# This is better than loading it on every request
with app.app_context():
    ml_utils.load_classifier_model()

@app.route('/')
def index():
    """Renders the main search page."""
    return render_template('index.html')

@app.route('/search', methods=['POST'])
def search():
    """Handles the search query submission."""
    query = request.form.get('query', '').strip()

    if not query:
        return redirect(url_for('index')) # Redirect back if query is empty

    print(f"\nReceived query: '{query}'")

    # 1. Predict Query Type using ML model
    query_type = ml_utils.predict_query_type(query)
    print(f"Predicted query type: {query_type}")

    # 2. Fetch Data from Sources based on query type
    all_results = []
    print("Fetching from UniProt...")
    uniprot_results = fetchers.fetch_uniprot(query, query_type)
    all_results.extend(uniprot_results)
    print(f"Found {len(uniprot_results)} results from UniProt.")

    print("Fetching from GenBank/NCBI...")
    genbank_results = fetchers.fetch_genbank(query, query_type)
    all_results.extend(genbank_results)
    print(f"Found {len(genbank_results)} results from GenBank/NCBI.")

    # Add calls to other fetchers (Ensembl, PDB) here if implemented
    # ensembl_results = fetchers.fetch_ensembl(query, query_type)
    # all_results.extend(ensembl_results)
    # pdb_results = fetchers.fetch_pdb(query, query_type)
    # all_results.extend(pdb_results)

    print(f"Total results before ranking: {len(all_results)}")

    # 3. Rank Results (using heuristic/ML)
    ranked_results = ml_utils.rank_results(all_results, query_type, query)
    print(f"Total results after ranking: {len(ranked_results)}")

    # 4. Render Results Page
    return render_template('results.html',
                           query=query,
                           query_type=query_type,
                           results=ranked_results)

if __name__ == '__main__':
    # Set debug=True for development (auto-reloads, provides debugger)
    # Set debug=False for production
    app.run(debug=True)