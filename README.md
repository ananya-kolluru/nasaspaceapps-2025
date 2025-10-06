# nasaspaceapps-2025

## Team SpaceGraph
Code for Space Biology Knowldge Engine using Python and Streamlit and developed using Gemini 2.5 Flash and Pro (free tiers).

This repository has code for the NASA Space Apps challenge 2025, for the project/theme: Build a Space Biology Knowledge Engine.

For this challenge, we have 2 working versions: one that used the CSV file provided at https://github.com/jgalazka/SB_publications/tree/main to fetch the individual abstracts of each publication for data analysis. In the next solution, we fetched publications from the NASA site https://genelab-data.ndc.nasa.gov/genelab/data/search using API calls. In both solutions,we fetched data into python lists, converted them to a Pandas Dataframe, then fed into a transformer model, to create embeddings. These were displayed in a basic local dashboard (hosted on http://localhost:8501/ of our working laptop) with search capability and a cluster graph with a configurable 'number of clusters (k)'.

For the first item, we downloaded the provided CSV file that contained titles and links to 608 publications. We then fetchced the actual abstracts of every publication into our local pandas dataframe and fed the abstracts to the keyword extraction program. This solution is in files data_preparation.py and app.py. All screenshots in the presentation demo made for the project are for this solution.

For the second solution (not our main focus), we  used Gemini to query NASA GeneLab API using JSON and Pandas Data frames to analyze the data. 

Once we had all the titles and abstracts in our dataframe, we ran it through Scikit-learn and used the model allenai/scibert_scivocab_uncased to get embeddings and transformed it to UMAP 2D shape. We completely relied in Gemini to come up with these models based on very specifc prompts to focus on space/scientific publications).

Once we have this dataframe, we can manipulate it to do different things like search all the descriptions fields (abstracts) of the published papers and run it through an ML logic to extract keywords. Then we use streamlit to run a simple web application to display the search engine and graphs.
