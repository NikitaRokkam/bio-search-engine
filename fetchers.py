# fetchers.py
import requests
from Bio import Entrez, SeqIO
import io # For handling string IO with SeqIO
import time

# IMPORTANT: Always provide your email to NCBI Entrez
Entrez.email = "bhaktid@iitbhilai.ac.in" # CHANGE THIS!

# --- UniProt Fetcher ---
UNIPROT_API_URL = "https://rest.uniprot.org/uniprotkb/search"
UNIPROT_BLAST_URL = "https://www.uniprot.org/blast" # Info page, actual submission is more complex

def fetch_uniprot(query, query_type):
    """Fetches data from UniProt KB."""
    results = []
    try:
        if query_type == "keyword":
            params = {
                "query": f'({query}) AND reviewed:true', # Search reviewed (Swiss-Prot) by default
                "format": "json",
                "size": 10 # Limit results for performance
            }
            response = requests.get(UNIPROT_API_URL, params=params, timeout=15)
            response.raise_for_status() # Raise error for bad status codes
            data = response.json()
            for entry in data.get("results", []):
                acc = entry.get("primaryAccession")
                prot_data = entry.get("proteinDescription", {})
                org_data = entry.get("organism", {})
                results.append({
                    "source": "UniProt",
                    "id": acc,
                    "title": prot_data.get("recommendedName", {}).get("fullName", {}).get("value", "N/A"),
                    "organism": org_data.get("scientificName", "N/A"),
                    "url": f"https://www.uniprot.org/uniprotkb/{acc}/entry",
                    "description": prot_data.get("submissionNames", [{}])[0].get("fullName", {}).get("value", None) # Alternative name
                })

        elif query_type == "sequence":
            # NOTE: UniProt REST API doesn't directly support sequence search (BLAST).
            # Proper BLAST requires submitting a job and polling results, which is complex for this simple example.
            # We will just add a placeholder message.
            # A workaround could be searching for short exact sequence matches, but less useful.
            print("UniProt sequence search (BLAST) requires job submission - skipping for this simple example.")
            results.append({
                "source": "UniProt (Info)",
                "id": "BLAST required",
                "title": "Sequence search requires UniProt BLAST tool",
                "organism": "",
                "url": UNIPROT_BLAST_URL,
                "description": "Use the UniProt website for sequence similarity searches."
             })
            # Simple exact match search (limited utility):
            # params = {
            #     "query": f'sequence:{query} AND reviewed:true',
            #     "format": "json", "size": 5
            # }
            # response = requests.get(UNIPROT_API_URL, params=params, timeout=15) ... process results ...


    except requests.exceptions.RequestException as e:
        print(f"Error fetching from UniProt: {e}")
    except Exception as e:
         print(f"An unexpected error occurred in UniProt fetcher: {e}")
    return results

# --- GenBank/NCBI Fetcher ---
def fetch_genbank(query, query_type):
    """Fetches data from NCBI (GenBank nucleotide and protein) using Entrez."""
    results = []
    db = "nucleotide" if query_type == "sequence" else "protein" # Default guess for sequence type
    MAX_RESULTS = 10

    try:
        # --- Keyword Search ---
        if query_type == "keyword":
            # Search both nucleotide and protein databases
            for current_db in ["nucleotide", "protein"]:
                handle = Entrez.esearch(db=current_db, term=query, retmax=MAX_RESULTS // 2)
                record = Entrez.read(handle)
                handle.close()
                ids = record.get("IdList", [])
                if not ids: continue

                # Fetch summaries for the found IDs
                summary_handle = Entrez.esummary(db=current_db, id=",".join(ids))
                summaries = Entrez.read(summary_handle)
                summary_handle.close()

                for summary in summaries:
                    results.append({
                        "source": f"GenBank ({current_db.capitalize()})",
                        "id": summary.get("AccessionVersion", "N/A"),
                        "title": summary.get("Title", "N/A"),
                        "organism": summary.get("Organism", "N/A"),
                        "url": f"https://www.ncbi.nlm.nih.gov/{current_db}/{summary.get('AccessionVersion', '')}",
                        "description": f"Length: {summary.get('Length', 'N/A')}, TaxID: {summary.get('TaxId', 'N/A')}"
                    })
                time.sleep(0.5) # Be nice to NCBI servers

        # --- Sequence Search (using BLAST via Entrez) ---
        elif query_type == "sequence":
            # Determine DB based on simple alphabet check (can be improved)
            query = query.upper()
            if all(c in 'ACGTUN' for c in query):
                db = "nucleotide"
                program = "blastn"
            else: # Assume protein
                db = "protein"
                program = "blastp"

            print(f"Running NCBI BLAST ({program} on {db}) for sequence...")
            # Entrez.qblast returns results directly (for shorter queries)
            # For long queries or production, use Biopython's NCBIWWW.qblast with job polling
            try:
                blast_handle = Entrez.qblast(program, db, query, hitlist_size=MAX_RESULTS)

                # Read results - this can take time!
                # NOTE: Entrez.read() can fail on complex BLAST XML. Parse manually if needed.
                # Using blast_records = NCBIXML.parse(blast_handle) is more robust but adds dependency on NCBIXML parser
                from Bio.Blast import NCBIXML # Import here as it's specific
                blast_records = NCBIXML.parse(blast_handle) # Use parser

                # Limit processing time / records
                count = 0
                for record in blast_records:
                     for alignment in record.alignments:
                        if count >= MAX_RESULTS: break
                        for hsp in alignment.hsps: # High-scoring pairs
                            results.append({
                                "source": f"NCBI BLAST ({db.capitalize()})",
                                "id": alignment.accession,
                                "title": alignment.title.split('>', 1)[-1].strip(), # Clean up title
                                "organism": "", # BLAST results don't always give organism easily here
                                "url": f"https://www.ncbi.nlm.nih.gov/{db}/{alignment.accession}",
                                "description": f"Score: {hsp.score}, E-value: {hsp.expect:.2e}, Identity: {hsp.identities}/{hsp.align_length} ({100*hsp.identities/hsp.align_length:.0f}%)"
                            })
                            count += 1
                            break # Usually take the best HSP per alignment
                     if count >= MAX_RESULTS: break

                blast_handle.close() # Important!
                if count == 0: print("No BLAST hits found.")

            except Exception as blast_err:
                print(f"NCBI BLAST query failed: {blast_err}")
                # Fallback: Maybe try an exact match search?
                print("Trying exact match search as fallback...")
                handle = Entrez.esearch(db=db, term=f'"{query}"[Sequence]', retmax=1) # Exact sequence search
                record = Entrez.read(handle)
                handle.close()
                ids = record.get("IdList", [])
                if ids:
                    summary_handle = Entrez.esummary(db=db, id=",".join(ids))
                    summaries = Entrez.read(summary_handle)
                    summary_handle.close()
                    for summary in summaries:
                         results.append({
                            "source": f"GenBank ({db.capitalize()}) Exact Match",
                            "id": summary.get("AccessionVersion", "N/A"),
                            "title": summary.get("Title", "N/A"),
                            "organism": summary.get("Organism", "N/A"),
                            "url": f"https://www.ncbi.nlm.nih.gov/{db}/{summary.get('AccessionVersion', '')}",
                            "description": f"Length: {summary.get('Length', 'N/A')}, TaxID: {summary.get('TaxId', 'N/A')}"
                         })


    except Exception as e:
        print(f"Error fetching from NCBI/GenBank: {e}")
        import traceback
        traceback.print_exc() # Print detailed error traceback

    return results

# --- Add Fetchers for Other Databases (Ensembl, PDB) Here ---
# e.g., def fetch_ensembl(query, query_type): ...
# e.g., def fetch_pdb(query, query_type): ...
# These would use their respective REST APIs (refer to their documentation)
# For simplicity, we'll stick to UniProt and GenBank for now.