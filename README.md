# nasaspaceapps-2025
Code for Space Biology Knowldge Engine
Team SpaceGraph

This repository has code for the NASA Space Apps challenge 2025, for the project/theme: Build a Space Biology Knowledge Engine.

For this challenge, we fetched publications from the NASA site https://genelab-data.ndc.nasa.gov/genelab/data/search using API calls, and store them in a Pandas Dataframe.

Once we have this dataframe, we can manipulate it to do different things like search all the descriptions fields (abstracts) of the published papers and run it through an ML logic to extract keywords. Then we use streamlit to run a simple web application to display the search engine and graphs.
