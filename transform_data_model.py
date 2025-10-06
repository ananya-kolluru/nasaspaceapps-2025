import pandas as pd
import numpy as np
import fastparquet as fp
import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
import umap.umap_ as umap
from umap.umap_ import UMAP # Older versions might require this- works
#from umap import UMAP       # Recommended for current versions


# Load the Parquet file
#print(f"Converting the parquet file, {PARQUET_FILE} to df")
#df_retrieved = fp.ParquetFile(PARQUET_FILE).to_pandas()

# Use sciBERT which needs to use Hugging Face Transformers
# make sure this was done: pip install torch transformers
# NOTE: You will need to install sentence-transformers for this simplified approach:
# pip install sentence-transformers
import torch
from sentence_transformers import SentenceTransformer

# 1. Load the SciBERT Sentence Model 🚀
# 'all-MiniLM-L6-v2' is a good balance of speed/accuracy for general use,
# but 'allenai/scibert_scivocab_uncased' is the domain-specific choice.
# We'll use a Sentence Transformer variant of SciBERT for easier document embedding.
# Note: Using the base model name 'all-MiniLM-L6-v2' is often practical for KeyBERT due to speed,
# but for domain-specific accuracy, SciBERT is technically superior.

# Using SciBERT via AutoModel is best for domain accuracy:
from transformers import AutoTokenizer, AutoModel

# SciBERT model specifically trained on scientific text
MODEL_NAME = 'allenai/scibert_scivocab_uncased' 
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModel.from_pretrained(MODEL_NAME)

def get_scibert_embedding(text):
    """Encodes a single document using SciBERT to get a document vector."""
    # 1. Tokenize the text
    inputs = tokenizer(
        text, 
        return_tensors='pt', 
        padding=True, 
        truncation=True, 
        max_length=512 # Max length for BERT models
    )
    
    # 2. Get the embeddings
    with torch.no_grad():
        outputs = model(**inputs)
    
    # 3. Extract the [CLS] token vector (the document representation)
    # The [CLS] token is the first token (index 0) in the sequence.
    cls_embedding = outputs.last_hidden_state[:, 0, :]
    
    # 4. Convert to a numpy array
    return cls_embedding.squeeze().numpy()


# Function to join the list of keywords into a single string
def join_keywords(keyword_list):
    # Check if the list is empty, and return an empty string if it is
    if not keyword_list:
        return ""
    # Join the keywords with a space (or comma-space) to create one large text
    return ' '.join(keyword_list) # Or ', '.join(keyword_list)

# Apply the joining function first, then apply the embedding function to the resulting string
# Note: For an empty string, the get_scibert_embedding function will still run, 
# but it will correctly return a zero-vector or an expected token for empty input, 
# preventing the index error.

# 2. Transform the Text Data
# Apply the function to your DataFrame column
print(df_retrieved.columns)  #print all the col names coz I keep getting 'keywords' missing error!

X_list = df_retrieved['keywords'].apply(join_keywords).apply(get_scibert_embedding).tolist()
print(df_retrieved[['accession', 'keywords']])

# Convert the list of vectors into a dense NumPy matrix
X = np.array(X_list)

# The result 'X' is now a dense embedding matrix
print(f"New SciBERT Embedding shape: {X.shape} - Dense (Documents, 768)")


# Initialize UMAP reducer
# n_neighbors: Controls how UMAP balances local vs. global structure.
# min_dist: Controls how tightly UMAP allows points to be packed.
print(umap.__file__)

reducer = umap.UMAP(
    n_components=2,          # Reduce to 2 dimensions for easy plotting
    n_neighbors=15,
    min_dist=0.1,
    random_state=42          # For reproducibility
)

# Fit and transform the data
# This process can be slow for very large datasets
embedding_2d = reducer.fit_transform(X)

print(f"UMAP 2D embedding shape: {embedding_2d.shape}")


# Create the scatter plot
plt.figure(figsize=(12, 10))
plt.scatter(
    embedding_2d[:, 0], # x-axis: first dimension
    embedding_2d[:, 1], # y-axis: second dimension
    s=5,                # Marker size
    alpha=0.8           # Transparency
)

plt.title('2D UMAP Projection of SciBERT Document Embeddings', fontsize=18)
plt.xlabel('UMAP Dimension 1', fontsize=14)
plt.ylabel('UMAP Dimension 2', fontsize=14)
plt.grid(True, linestyle='--', alpha=0.6)
plt.show()


from sklearn.cluster import KMeans

# Determine the number of clusters (k) - often determined by elbow method or context
k = 5
kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
# Fit K-Means on the 2D data
clusters = kmeans.fit_predict(embedding_2d)
cluster_labels = kmeans.fit_predict(embedding_2d)

# Add the cluster labels to the main dataset with titles/abstracts
df_retrieved['cluster'] = cluster_labels

n_clusters = k
# Each doc has a cluster number (0, 1, 2..) Loop through each cluster and print titles to see similarity
for i in range(n_clusters):
    # Filter the DataFrame to include only the documents in the current cluster
    cluster_data = df_retrieved[df_retrieved['cluster'] == i]

    print(f"\n--- Cluster {i} ({len(cluster_data)} documents) ---")
    
    # Print the titles and/or keyphrases for the first few documents
    # You can increase the '.head(10)' number if you need to inspect more.
    for index, row in cluster_data.head(10).iterrows():
        print(f"  Title: {row['title']}")
        
        # This line assumes you have a 'keyphrases' column from KeyBERT
        if 'keywords' in row and row['keywords'] is not None:
             # Convert list of tuples/strings to a single readable string
             keyphrase_str = ", ".join([str(k) for k in row['keywords']])
             print(f"  Keyphrases: {keyphrase_str}")
             
    # After reviewing the printed titles and keyphrases, you manually assign a label
    # Example: If all titles are about heart failure, you write that down.
    
    # Manually assigned label for this cluster (you fill this in!)
    # print(f"\n>>>> MANUAL LABEL: '...")

# Plot the colored scatter plot
plt.figure(figsize=(12, 10))
scatter = plt.scatter(
    embedding_2d[:, 0],
    embedding_2d[:, 1],
    c=clusters,          # Color points by cluster label
    cmap='Spectral',     # Color map to use
    s=10,
    alpha=0.9
)

# Add a color bar and legend
plt.title(f'2D UMAP Projection with K-Means (k={k}) Clusters', fontsize=18)
plt.xlabel('UMAP Dimension 1', fontsize=14)
plt.ylabel('UMAP Dimension 2', fontsize=14)

# Create legend handles for the clusters
legend1 = plt.legend(*scatter.legend_elements(),
                    loc="lower left", title="Clusters")
plt.gca().add_artist(legend1)
plt.grid(True, linestyle='--', alpha=0.6)
plt.show()
