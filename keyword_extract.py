#  Initialize KeyBERT Model
# -------------------------------------------------
# This will download a pre-trained model for keyword extraction.
# This might take a moment on the first run.
from keybert import KeyBERT

#MODEL_NAME = 'astrobert/astrobert_astropub_uncased' # This is more specialized, but ge hugging face 401 error - needs authenticated API?
MODEL2_NAME = 'allenai/scibert_scivocab_uncased'

print("\nInitializing KeyBERT to use a Space domain specific model...")
kw_model = KeyBERT(model=MODEL2_NAME)
print(f"KeyBERT model initialized to use {MODEL2_NAME} successfully.")

# This function will take a text and use the KeyBERT model to
# extract relevant keywords.


def extract_keywords(text):
    """
    Extracts keywords from a given text using the KeyBERT model.
    
    Args:
        text (str): The input text (e.g., an abstract).
        
    Returns:
        list: A list of extracted keywords (as strings).
    """
    try:
        # We use keyphrase_ngram_range to get single words and two-word phrases.
        # stop_words='english' helps remove common, non-descriptive words.
        # top_n specifies the number of top keywords to return.
        keywords = kw_model.extract_keywords(
            text,
            keyphrase_ngram_range=(1, 2),
            stop_words='english',
            top_n=20
        )
        # The model returns keywords with scores; we only want the words.
        return [keyword for keyword, score in keywords]
    except Exception as e:
        print(f"Could not extract keywords. Error: {e}")
        return []

 We will now apply our function to the 'description' column of our DataFrame.
# This may take some time depending on the number of studies fetched.

if not df_retrieved.empty:
    print("\nExtracting keywords from study descriptions...")
    # The .apply() method iterates over each row in the 'description' column
    # and runs our function on it.df_retrieved['description']
    df_retrieved['keywords'] = df_retrieved['description'].apply(extract_keywords)

    # Display the DataFrame with the new 'keywords' column
    print("\nDataFrame with extracted keywords:")
    print(df_retrieved[['accession', 'keywords']]) #print everything for now (top_n=20 above); change to .head() for top5
else:
    print("\nSkipping keyword extraction because no study data was loaded.")

#Just printing to make sure we have the df_retrieved dataframe:
print(df_retrieved[['accession', 'keywords']])

print("\nKeyword extraction function is ready.")

if not df_retrieved.empty:
    print("\n--- Sample of OSDR Studies with Extracted Keywords ---")
    
    # Display the first 5 entries
    for index, row in df_retrieved.head(5).iterrows():
        print(f"\n✅ Study Accession: {row['accession']}")
        
        # Pretty print the description
        print("\n   Description:")
        wrapped_description = textwrap.fill(row['description'], width=80, initial_indent='   ', subsequent_indent='   ')
        print(wrapped_description)
        
        # Print keywords
        # TBD: What to do with the score of keywords? sort by score etc.. Need to clean up, keywords like bone/bones/process bone/etc..
        print(f"\n   🔑 Extracted Keywords: {', '.join(row['keywords'])}")
        print("-" * 50)
else:
    print("\nCannot display results as no data was loaded.")

