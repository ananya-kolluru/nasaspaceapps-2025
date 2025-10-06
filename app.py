import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import umap
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import json # Added for handling JSON-encoded keywords

# Define file paths
CSV_FILE_STREAMIT = 'nasa_studies_data.csv' # Kept for potential future use, but PARQUET is preferred
PARQUET_FILE_STREAMLIT = 'nasa_studies_data.parquet'
EMBEDDINGS_FILE = 'scibert_embeddings.npy'

# --- 1. DATA LOADING AND CACHING (FAST RELOADS) ---

@st.cache_data
def load_data():
    """
    Loads the processed data and embeddings.
    """
    try:
        df_retrieved = pd.read_parquet(PARQUET_FILE_STREAMLIT)
        
        # Ensure the 'keywords' column is treated as a list of strings
        df_retrieved['keywords'] = df_retrieved['keywords'].apply(
            # We use json.loads because 'data_preparation.py' encodes the list as a JSON string
            lambda x: json.loads(x) if isinstance(x, str) else (x if isinstance(x, list) else [])
        )
    except FileNotFoundError:
        st.error("Error: Could not find 'nasa_studies_data.parquet'. Please run data_preparation.py to generate it.")
        return pd.DataFrame(), np.array([])
    except Exception as e:
        st.error(f"Error loading Parquet file: {e}")
        return pd.DataFrame(), np.array([])


    try:
        X = np.load(EMBEDDINGS_FILE)
        if X.shape[0] != len(df_retrieved):
            st.error(f"Error: Embeddings shape ({X.shape[0]}) does not match DataFrame rows ({len(df_retrieved)}). Check your saving process.")
            return pd.DataFrame(), np.array([])
    except FileNotFoundError:
        st.error("Error: Could not find 'scibert_embeddings.npy'. Please run data_preparation.py to generate it.")
        return df_retrieved, np.array([])

    return df_retrieved, X

# --- 2. SEMANTIC VISUALIZATION AND CLUSTERING ---

def generate_visualization(X: np.ndarray, df: pd.DataFrame, n_clusters: int, cluster_labels: dict):
    """
    Applies UMAP to reduce dimensions and KMeans for clustering.
    Returns a Matplotlib figure and the cluster assignments.
    """
    if X.size == 0:
        return None, []

    # 1. Dimensionality Reduction using UMAP
    reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, n_components=2, random_state=42)
    embedding = reducer.fit_transform(X)
    
    df['UMAP-1'] = embedding[:, 0]
    df['UMAP-2'] = embedding[:, 1]

    # 2. Clustering using KMeans
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(X)
    df['Cluster'] = clusters

    # 3. Visualization with Matplotlib
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Define color map for clusters
    cmap = plt.cm.get_cmap('Spectral', n_clusters)
    
    scatter = ax.scatter(
        df['UMAP-1'],
        df['UMAP-2'],
        c=df['Cluster'],
        cmap=cmap,
        s=10,
        alpha=0.7,
        label=df['Cluster']
    )
    
    # Add cluster centroids (optional, but helpful)
    for cluster_id in range(n_clusters):
        centroid_x = df[df['Cluster'] == cluster_id]['UMAP-1'].mean()
        centroid_y = df[df['Cluster'] == cluster_id]['UMAP-2'].mean()
        
        # Display the cluster label at the centroid
        label = cluster_labels.get(cluster_id, f"Cluster {cluster_id}")
        ax.annotate(
            label,
            (centroid_x, centroid_y),
            textcoords="offset points",
            xytext=(0, 10),
            ha='center',
            fontsize=8,
            fontweight='bold',
            color=cmap(cluster_id)
        )


    ax.set_title(f'UMAP Projection of {len(df)} Studies (k={n_clusters})')
    ax.set_xlabel('UMAP Dimension 1')
    ax.set_ylabel('UMAP Dimension 2')
    
    # Create a legend
    legend1 = ax.legend(*scatter.legend_elements(), 
                        loc="lower left", 
                        title="Clusters", 
                        bbox_to_anchor=(1.05, 0))
    ax.add_artist(legend1)

    plt.tight_layout()
    return fig, clusters


# --- 3. KEYWORD LABELING FUNCTION ---

@st.cache_data
def generate_cluster_labels(df: pd.DataFrame, clusters: np.ndarray, n_clusters: int) -> dict:
    """
    Generates representative keyword labels for each cluster using TF-IDF.
    """
    # Ensure all data has been clustered
    df_clustered = df.copy()
    df_clustered['Cluster'] = clusters
    
    cluster_labels = {}
    
    for i in range(n_clusters):
        cluster_data = df_clustered[df_clustered['Cluster'] == i]
        
        if cluster_data.empty:
            cluster_labels[i] = "No studies"
            continue

        # Combine all title and abstract texts for the current cluster
        text_data = cluster_data['title'] + " " + cluster_data['abstract']
        
        # Use TF-IDF to find important terms within the cluster
        vectorizer = TfidfVectorizer(stop_words='english', max_features=50)
        tfidf_matrix = vectorizer.fit_transform(text_data)
        feature_names = vectorizer.get_feature_names_out()
        
        # Calculate the mean TF-IDF score for each term in the cluster
        avg_tfidf = np.mean(tfidf_matrix.toarray(), axis=0)
        
        # Get the top 3 terms
        top_indices = avg_tfidf.argsort()[-3:]
        top_terms = [feature_names[j] for j in reversed(top_indices)]
        
        cluster_labels[i] = ", ".join(term.capitalize() for term in top_terms)
        
    return cluster_labels

# --- 4. SEARCH FUNCTION ---

def search_studies_by_keyword(df: pd.DataFrame, query: str):
    """
    Filters studies based on a search query applied to title, abstract, and keywords.
    """
    if not query:
        return df

    # Convert query to lowercase for case-insensitive search
    query_lower = query.lower()

    # Create boolean masks for each searchable column
    title_match = df['title'].str.lower().str.contains(query_lower, na=False)
    abstract_match = df['abstract'].str.lower().str.contains(query_lower, na=False)
    
    # Check if the query is in any of the list elements in the 'keywords' column
    # Ensure keywords are handled correctly (they are lists of strings after load_data)
    keyword_match = df['keywords'].apply(lambda kws: any(query_lower in kw.lower() for kw in kws))

    # Combine the masks
    filtered_df = df[title_match | abstract_match | keyword_match]
    return filtered_df

# --- 5. MAIN STREAMLIT APP LAYOUT ---

def main():
    st.set_page_config(layout="wide", page_title="NASA Study Explorer")

    st.title("SpaceGraph: NASA Space Biology Study Explorer")
    st.markdown("Use SciBERT-powered semantic search and clustering to explore relevant space biology research.")
    st.markdown("---")

    # Load data once and cache the result
    df_retrieved, X = load_data()

    if df_retrieved.empty or X.size == 0:
        return

    # --- 1. CLUSTERING PARAMETERS (Sidebar) ---
    st.sidebar.header("Clustering Settings")
    
    # Slider for number of clusters (k)
    k_value = st.sidebar.slider(
        'Select Number of Clusters (k):', 
        min_value=2, 
        max_value=15, 
        value=5, 
        step=1
    )
    
    # Re-run clustering and labeling based on new k-value
    # Note: Visualization function applies UMAP and KMeans to X
    fig, clusters = generate_visualization(X, df_retrieved, n_clusters=k_value, cluster_labels={})

    if fig:
        df_retrieved['Cluster'] = clusters
        
        # Generate human-readable labels for the clusters
        cluster_labels = generate_cluster_labels(df_retrieved, clusters, k_value)

    # --- 2. SEARCH AND FILTER SECTION ---
    
    col_search, col_filter = st.columns([2, 1])

    with col_search:
        search_query = st.text_input(
            "Search Studies:",
            placeholder="e.g., radiation effects, bone loss, Mars"
        )
    
    with col_filter:
        # Create multiselect options using the generated labels
        cluster_options = sorted(df_retrieved['Cluster'].unique())
        selected_clusters = st.multiselect(
            "Filter by Cluster Theme:",
            options=cluster_options,
            format_func=lambda x: f"Cluster {x}: {cluster_labels.get(x, 'N/A')}"
        )

    # Apply filters
    filtered_df = search_studies_by_keyword(df_retrieved, search_query)
    if selected_clusters:
        filtered_df = filtered_df[filtered_df['Cluster'].isin(selected_clusters)]

    st.subheader(f"Results ({len(filtered_df)} studies found)")

    # NEW: Improved display using st.expander for details
    if filtered_df.empty:
        st.warning("No studies match your criteria.")
    else:
        for _, row in filtered_df.head(50).iterrows(): # Display up to 50 results
            # Ensure 'Cluster' exists before attempting to access cluster_labels
            cluster_id = row.get('Cluster', 'N/A')
            cluster_title = cluster_labels.get(cluster_id, 'N/A')

            with st.expander(f"**{row['link']}**: {row['title']}"):
                st.markdown(f"**Cluster:** {cluster_id} - *{cluster_title}*")
                
                # Display keywords in a clean format
                keywords_list = row['keywords'] # This is already a list of strings
                if keywords_list:
                    st.markdown("**Keywords:**")
                    # Create columns for better layout
                    kw_cols = st.columns(4)
                    for i, kw in enumerate(keywords_list[:12]): # Show up to 12 keywords
                        kw_cols[i % 4].markdown(f"- `{kw.strip()}`")
                
                if 'description' in row and pd.notna(row['description']) and row['description']:
                    st.markdown(f"**Description:** {row['description']}")

                st.markdown(f"**Abstract:** {row['abstract']}")

                # Ensure 'link' is checked and displayed if available
                if 'link' in row and pd.notna(row['link']):
                    st.markdown(f"[View Full Publication]({row['link']})")


    # 3. VISUALIZATION DASHBOARD (Placed here, below the results)
    st.markdown("---")
    st.header("Semantic Map of Studies (UMAP)")
    st.markdown("<p>Visualize the semantic relationships between all loaded studies.</p>", unsafe_allow_html=True)
    
    # We already have the figure and clusters from the sidebar setup, just plot it
    if fig:
        st.pyplot(fig)
    
    

if __name__ == "__main__":
    main()
