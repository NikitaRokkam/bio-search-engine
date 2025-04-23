# Unified Bio Search Engine (with ML Query Classifier)

A simple web app to search UniProt and GenBank/NCBI, using **Machine Learning** to understand your query type.

## Features

*   Unified search for UniProt & NCBI GenBank.
*   Accepts sequences or keywords.
*   **ML-Powered Query Classification:** Uses a **Scikit-learn** (TF-IDF + Naive Bayes) model to automatically predict if your input is a 'sequence' or 'keyword'.
*   **Adaptive Search:** Uses the ML prediction to choose the right search strategy (keyword search vs. BLAST sequence search).
*   Displays ranked results with links.

## Tech Stack

*   Python 3, Flask
*   **Machine Learning:** Scikit-learn, Pandas, Joblib
*   **Bioinformatics:** Biopython
*   HTTP: Requests
*   HTML, CSS

## Quick Setup

1.  **Clone:**
    ```bash
    git clone <your-repository-url>
    cd bio_search_engine
    ```
2.  **Create & Activate Virtual Environment:**
    ```bash
    # Windows
    python -m venv venv
    .\venv\Scripts\activate

    # Linux/macOS
    python3 -m venv venv
    source venv/bin/activate
    ```
3.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
4.  **Set NCBI Email:**
    *   Open `fetchers.py`.
    *   Change `Entrez.email = "your.email@example.com"` to **your actual email**.

5.  **Train ML Classifier Model (Run Once):**
    ```bash
    python train_classifier.py
    ```
    *(This creates the `ml_models/query_classifier_pipeline.joblib` file)*

## Run the App

1.  Make sure your virtual environment is active (`(venv)` prefix).
2.  Start the Flask server:
    ```bash
    python app.py
    ```
3.  Open your browser to `http://127.0.0.1:5000`.

## Usage

1.  Go to the app's URL.
2.  Enter a DNA/protein sequence or keyword.
3.  Click "Search".
4.  The app uses its ML model to classify the query and fetches results accordingly. View the ranked results.
