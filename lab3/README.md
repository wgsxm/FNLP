# Dependency
- sentence_transformers
- torch
- rank_bm25
- tqdm
# Code Structure
## functions
- `predict_one_sample`: predict the most similar sentence in the corpus for the claim for a sample
- `predict`: predict the most similar sentence in the corpus for each claim for a dataset
- 'recall': calculate the recall of the model
## main
- load the json file
- Part 2.1
    - build a dense retrieval model from sentence-transformers
    - predict for the development set and test set
    - calculate the recall and save the result
- Part 2.3
    - use the BM25 model to predict for the development set
    - calculate the recall