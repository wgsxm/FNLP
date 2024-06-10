# build a sentence evidence retriever
from typing import *
import json

from sentence_transformers import SentenceTransformer
import torch
from tqdm import tqdm
from rank_bm25 import BM25Okapi

def predict_one_sample(retriever ,sample: Dict, top_k: int = 5) -> Dict:
    sentences = []
    sentences_dict = {}
    for i in range(len(sample['candidates'])):
        sentences.append(sample['candidates'][i][1])
        sentences_dict[i] = sample['candidates'][i][0]
    embeddings = retriever.encode(sentences, device='cuda')
    claim_embedding = retriever.encode([sample['claim']], device='cuda')
    similarities = retriever.similarity(claim_embedding, embeddings).reshape(-1)
    top_k = min(top_k, len(similarities))
    top_k_indices = torch.topk(similarities, top_k).indices.cpu().numpy()
    return {
        'id': sample['id'],
        'prediction': [sentences_dict[i] for i in top_k_indices]
    }
def predict(retriever, data: List[Dict], top_k: int = 5) -> List[Dict]:
    predictions = []
    for sample in tqdm(data):
        predictions.append(predict_one_sample(retriever, sample, top_k))
    return predictions
def recall(predictions: List[Dict], data: List[Dict]) -> float:
    recall = 0
    for i in range(len(data)):
        TP = 0
        for pred in predictions[i]['prediction']:
            if pred in data[i]['evidence']:
                TP += 1
        recall += TP / len(data[i]['evidence'])
    return recall / len(data)

if __name__ == '__main__':
    
    # load the json file
    data_folder = 'FNLP_lab_3_data'
    dev_data = json.load(open(f'{data_folder}/dev.json'))
    test_data = json.load(open(f'{data_folder}/test.json'))
    print('Data loaded.')
    
    # Part 2.1
    # build a dense retriever
    dense_retriever = SentenceTransformer('all-MiniLM-L6-v2').to('cuda')

    # predict on dev data and test data
    print('Predicting on dev data...(DENSE RETRIEVER)')
    dev_pred = predict(dense_retriever, dev_data, 5)
    print('Predicting on test data...(DENSE RETRIEVER)')
    test_pred = predict(dense_retriever, test_data, 5)
    
    # evaluate the model
    print(f'Recall on development set for dense retriever: {recall(dev_pred, dev_data)}')
    
    # save the predictions
    json.dump(dev_pred, open('dev_predictions.json', 'w'), indent=2)
    json.dump(test_pred, open('test_predictions.json', 'w'), indent=2)
    
    # Part 2.3
    # build a sparse retriever and predict on dev data
    print('Predicting on dev data...(SPARSE RETRIEVER)')
    dev_pred_sparse = []
    for sample in tqdm(dev_data):
        tokenized_sentences = []
        sentences_dict = {}
        for i in range(len(sample['candidates'])):
            tokenized_sentences.append(sample['candidates'][i][1].split(" "))
            sentences_dict[i] = sample['candidates'][i][0]
        bm25 = BM25Okapi(tokenized_sentences)
        query = sample['claim']
        tokenized_query = query.split(" ")
        scores = bm25.get_scores(tokenized_query)
        topk = min(5, len(scores))
        top_k_indices = torch.topk(torch.tensor(scores), topk).indices.cpu().numpy()
        dev_pred_sparse.append({
            'id': sample['id'],
            'prediction': [sentences_dict[i] for i in top_k_indices]
        })
    
    # evaluate the model
    print(f'recall on development set for sparse retriever: {recall(dev_pred_sparse, dev_data)}')