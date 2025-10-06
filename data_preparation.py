import pandas as pd
import requests
from bs4 import BeautifulSoup
import numpy as np
import os
import time
# *** ADDED JSON IMPORT ***
import json
from sklearn.feature_extraction.text import TfidfVectorizer

# --- NEW IMPORTS FOR SCI-BERT EMBEDDING GENERATION ---
# You must install these libraries: pip install transformers torch
from transformers import AutoTokenizer, AutoModel
import torch
# *******************************************************

# --- CONFIGURATION ---
# REMOVED CSV_URL - Now using a local file path
LOCAL_CSV_FILE = "SB_publication_PMC.csv" #download and use CSV file locally
PARQUET_OUTPUT = 'nasa_studies_data.parquet'
EMBEDDINGS_OUTPUT = 'scibert_embeddings.npy'
# Set a delay to avoid overwhelming the server (be polite when scraping)
SCRAPING_DELAY = 0.5 
# Use GPU if available
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- ACTUAL EMBEDDING GENERATION FUNCTION ---
def generate_scibert_embeddings(df: pd.DataFrame) -> np.ndarray:
    """
    Loads SciBERT, tokenizes the combined 'title' and 'abstract' text, 
    and generates mean-pooled embeddings for clustering.
    """
    # 1. Check if the pre-existing, correctly sized embeddings file exists
    if os.path.exists(EMBEDDINGS_OUTPUT) and np.load(EMBEDDINGS_OUTPUT).shape[0] == len(df):
        print(f"\nLoading existing and aligned embeddings file: {EMBEDDINGS_OUTPUT}")
        return np.load(EMBEDDINGS_OUTPUT)

    print("\n--- Starting SciBERT Embedding Generation ---")

    # 2. Load Model and Tokenizer
    model_name = 'allenai/scibert_scivocab_uncased'
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModel.from_pretrained(model_name).to(DEVICE)
        model.eval() # Set model to evaluation mode
    except Exception as e:
        # --- ENHANCED ERROR LOGGING ---
        print("\n--- FATAL EMBEDDING ERROR ---")
        print("SciBERT model loading FAILED. This is usually due to missing libraries or internet issues.")
        print("Please ensure you have run: pip install transformers torch")
        print(f"Original Error: {e}")
        print("----------------------------")
        print(f"Falling back to DUMMY data ({len(df)} studies will have zero-embeddings). App results will be meaningless until this is fixed.")
        # -----------------------------
        dummy_embedding_dim = 768 
        return np.zeros((len(df), dummy_embedding_dim))
    
    # 3. Prepare Input Text (Combine title and abstract)
    texts = df['title'] + " " + df['abstract']
    
    all_embeddings = []
    batch_size = 16 # Adjust based on your available memory
    
    with torch.no_grad(): # Disable gradient calculation for efficiency
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size].tolist()
            
            # 4. Tokenization
            encoded_input = tokenizer(
                batch_texts, 
                padding=True, 
                truncation=True, 
                return_tensors='pt', 
                max_length=512
            ).to(DEVICE)
            
            # 5. Generate Embeddings (Model Forward Pass)
            model_output = model(**encoded_input)
                
            # Get the token embeddings (last hidden state)
            token_embeddings = model_output.last_hidden_state
            
            # Calculate mean-pooled embeddings (average across all tokens)
            mask = encoded_input['attention_mask'].unsqueeze(-1).expand(token_embeddings.size()).float()
            sum_embeddings = torch.sum(token_embeddings * mask, 1)
            sum_mask = torch.clamp(mask.sum(1), min=1e-9)
            mean_pooled_embeddings = sum_embeddings / sum_mask
            
            all_embeddings.append(mean_pooled_embeddings.cpu().numpy())
            
            if (i + batch_size) % 100 == 0:
                print(f"   Generated embeddings for {min(i + batch_size, len(df))}/{len(df)} studies.")

    # 6. Concatenate and Return
    X = np.vstack(all_embeddings)
    print(f"Embedding generation complete. Shape: {X.shape}")
    return X


# --- SCRAPING FUNCTION ---

def scrape_abstracts(url: str) -> str:
    """
    Fetches the content from a URL and attempts to extract the abstract text using
    multiple robust selectors, including a longest-text fallback.
    """
    try:
        # User-Agent to mimic a browser, preventing a 403 Forbidden error
        headers = {'User-Agent': 'Mozilla/5.0'}
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # --- 1. PRIMARY TARGETED SELECTORS ---
        abstract_tags_priority = [
            # Specific Section IDs/Classes
            soup.find('div', class_='abstract'), 
            soup.find('section', {'id': 'abstract'}), 
            soup.find('div', id='abs1'),
            
            # Common Publisher Classes (e.g., ScienceDirect, Wiley, Springer)
            soup.find('div', class_='abstract-text'), 
            soup.find('div', class_='article-section__content'), 
            soup.find('section', class_='article-section abstract'),
            soup.find('div', class_='core-abstract'),
            soup.find('section', {'role': 'doc-abstract'}), # ARIA role for abstract
        ]
        
        # Check priority tags
        for tag in abstract_tags_priority:
            if tag:
                abstract_text = tag.get_text(separator=' ', strip=True)
                # Remove common header text like "Abstract"
                abstract_text = abstract_text.replace('Abstract', '', 1).replace('SUMMARY', '', 1).strip()
                if len(abstract_text) > 50: 
                    return abstract_text

        # --- 2. FALLBACK: SEARCH FOR CONTENT FOLLOWING AN 'ABSTRACT' HEADER ---
        # Look for headers containing the word 'abstract' or 'summary'
        abstract_headers = soup.find_all(['h2', 'h3', 'h4'], string=lambda t: t and ('abstract' in t.lower() or 'summary' in t.lower()))
        
        for header in abstract_headers:
            # Look at the immediate next sibling element (the content block right after the header)
            next_sibling = header.find_next_sibling()
            if next_sibling and next_sibling.name in ['div', 'p', 'section']:
                 abstract_text = next_sibling.get_text(separator=' ', strip=True)
                 if len(abstract_text) > 50:
                     return abstract_text
                     
        # --- 3. LAST RESORT: LONGEST PARAGRAPH HEURISTIC ---
        all_paragraphs = soup.find_all('p')
        longest_p = ""
        
        for p in all_paragraphs:
            text = p.get_text(strip=True)
            # Only consider paragraphs long enough and not copyright/metadata
            if len(text) > 150 and not any(phrase in text.lower() for phrase in ['copyright', 'all rights reserved', 'received', 'published online']):
                if len(text) > len(longest_p):
                    longest_p = text
                    
        if len(longest_p) > 50:
             return longest_p
             
    except requests.exceptions.RequestException:
        pass
    except Exception:
        pass

    return "" # Return empty string if abstract is not found or error occurs

def extract_abstract_from_pmc_html(html):
    soup = BeautifulSoup(html, "html.parser")
    # Find heading with “Abstract”
    hdr = soup.find(lambda tag: tag.name in ["h2", "h3"] and "Abstract" in tag.get_text())
    if hdr:
        # Get following <p> siblings until next heading or section
        parts = []
        for sibling in hdr.find_next_siblings():
            if sibling.name in ["h2", "h3", "h4"]:
                break
            if sibling.name == "p":
                parts.append(sibling.get_text(" ", strip=True))
        if parts:
            return " ".join(parts)
    # Fallback: old selectors
    for sel in ["div.abstract p", "section#abstract p", "article .abstract p", "div#abstract p"]:
        el = soup.select_one(sel)
        if el:
            return el.get_text(" ", strip=True)
    return ""


def prepare_data_from_csv():
    """
    Reads from the local CSV file, cleans, scrapes abstracts, extracts keywords, and saves the data.
    """
    print(f"1. Reading data from local file: {LOCAL_CSV_FILE}")
    try:
        # Read from the local file path
        df = pd.read_csv(LOCAL_CSV_FILE)
    except FileNotFoundError:
        print(f"Error: Local CSV file '{LOCAL_CSV_FILE}' not found. Please ensure it is uploaded.")
        return
    except Exception as e:
        print(f"Error reading local CSV: {e}")
        return

    print(f"2. Initial studies loaded: {len(df)}")
    
    # --- 3. COLUMN RENAMING & CLEANING ---
    
    # Diagnostic print to check if the 'Keywords' column is present in the raw data
    print(f"   Initial columns: {df.columns.tolist()}")

#    column_mapping = {
#        'Accession': 'accession', # there is no accession or ID in the CSV file!
#        'Title': 'title',
#        'Link': 'link',         
#        'PMC ID': 'accession',  
#        'Keywords': 'keywords', # Target keyword column name
#    }
    
    column_mapping = {
        'Title': 'title',
        'Link': 'link',         
        'PMC ID': 'accession',  
        'Keywords': 'keywords', # Target keyword column name
    }
    df.rename(columns={k: v for k, v in column_mapping.items() if k in df.columns}, inplace=True)
    
    # Ensure 'link' column exists and is used for scraping -- TBD remove
    if 'link' not in df.columns and 'URL' in df.columns:
        df.rename(columns={'URL': 'link'}, inplace=True)
        
    if 'link' not in df.columns:
        print("Fatal Error: Could not find 'Link' or 'URL' column for scraping.")
        return

    # Prepare columns for text data
    df['description'] = ''
    df['abstract'] = '' 
    
    text_cols = ['title', 'abstract', 'description']
    df[text_cols] = df[text_cols].fillna('')
    
    # --- 4. SCRAPE ABSTRACTS ---
    print("\n3. Starting web scraping for abstracts (This will take a few minutes)...")
    abstracts = []
    
    for i, row in df.iterrows():
        url = row['link']
        #abstract = scrape_abstracts(url)
        # Fetch the abstract from the URL/link in the CSV file
        rsp = requests.get(url, headers={"User-Agent": "Mozilla/5.0"})
        text = extract_abstract_from_pmc_html(rsp.text)
        #print("Extracted abstract:", text[:4000]) # commented out, extraction works
        abstracts.append(text)
        
        # Print progress and introduce polite delay
        if (i + 1) % 50 == 0 or (i + 1) == len(df):
             print(f"   Processed {i + 1}/{len(df)} studies.")
        time.sleep(SCRAPING_DELAY) 
        
    df['abstract'] = abstracts
    print("   Scraping complete.")
    
    # --- 5. KEYWORD EXTRACTION LOGIC ---
    
    print("\n4. Processing and extracting keywords...")

    def process_keywords(kw_str):
        if pd.isna(kw_str) or not kw_str:
            return []
        if isinstance(kw_str, str):
            if ';' in kw_str:
                return [k.strip() for k in kw_str.split(';') if k.strip()]
            else:
                 return [k.strip() for k in kw_str.split(',') if k.strip()]
        return []

    # Initialize keywords column if not present (i.e., if 'Keywords' was not in the original CSV)
    if 'keywords' not in df.columns:
        df['keywords'] = [[]] * len(df)
    else:
        # Apply the parsing logic if the column is present
        df['keywords'] = df['keywords'].apply(process_keywords)
    
    # --- 6. DUPLICATE REMOVAL ---
    initial_count = len(df)
    if 'accession' in df.columns and df['accession'].any():
        df.drop_duplicates(subset=['accession'], keep='first', inplace=True)
        print(f"   Removed {initial_count - len(df)} duplicate studies based on accession.")
    else:
        df.drop_duplicates(subset=['link'], keep='first', inplace=True)
        print("   Warning: 'accession' column missing or empty, using 'link' for duplicate removal.")

    # --- 7. GENERATE EMBEDDINGS (CRUCIAL STEP) ---
    
    df.reset_index(drop=True, inplace=True) 
    
    # Filter out studies where scraping failed to get an abstract
    df_cleaned = df[df['abstract'].str.len() > 50].copy()
    print(f"   Filtered down to {len(df_cleaned)} studies with valid abstracts for embedding.")
    
    # --- NEW: KEYWORD FALLBACK AFTER CLEANING ---
    # Check if the resulting 'keywords' column is empty across the entire cleaned dataset
    if df_cleaned['keywords'].apply(len).sum() == 0:
        print("   Warning: Keywords missing or empty after cleaning. Generating fallback keywords from abstract/title.")
        
        texts = df_cleaned['title'] + " " + df_cleaned['abstract']
        
        # 1. Fit TF-IDF on the combined text of the cleaned studies
        tfidf_vectorizer = TfidfVectorizer(stop_words='english', max_features=1000)
        tfidf_matrix = tfidf_vectorizer.fit_transform(texts)
        feature_names = tfidf_vectorizer.get_feature_names_out()
        
        # 2. Extract Top 5 most important terms for each study
        fallback_keywords = []
        for i in range(tfidf_matrix.shape[0]):
            scores = tfidf_matrix[i, :].toarray().flatten()
            # Get indices of top 5 terms
            top_term_indices = scores.argsort()[-5:] 
            top_terms = feature_names[top_term_indices]
            fallback_keywords.append(top_terms.tolist())
            
        df_cleaned['keywords'] = fallback_keywords
        print("   Fallback keyword generation complete.")


    X = generate_scibert_embeddings(df_cleaned)
    
    # --- 8. FINAL SAVE ---
    
    # CRITICAL FIX: Convert the list of keywords to a stable JSON string for Parquet serialization
    df_cleaned['keywords'] = df_cleaned['keywords'].apply(lambda x: json.dumps(x))

    df_cleaned.to_parquet(PARQUET_OUTPUT, index=False)
    print(f"\n5. Cleaned metadata saved to {PARQUET_OUTPUT}")
    
    np.save(EMBEDDINGS_OUTPUT, X)
    print(f"6. Embeddings saved to {EMBEDDINGS_OUTPUT}")
    
    print("\nSUCCESS: Data files are ready for the Streamlit app.")

if __name__ == "__main__":
    prepare_data_from_csv()
