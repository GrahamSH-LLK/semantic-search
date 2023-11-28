from sentence_transformers import SentenceTransformer, util
import torch
import json
import pickle
from flask import Flask, request, jsonify

embedder = SentenceTransformer('trained_model')
# populate corpus from json file
import json
corpus = []
corpus_embeddings = []
try:
    with open('embeddings.pkl', "rb") as fIn:
        # check if file exists


        stored_data = pickle.load(fIn)
        corpus = stored_data['sentences']
        corpus_embeddings = stored_data['embeddings']
except:
    corpus = []
    with open('corpus.json', 'r') as f:
        corpus_json = json.load(f)
    # print(corpus[0].text)
        # iterate over corpus object and push to list
        for i in corpus_json:
            corpus.append({'text': corpus_json[i]["textContent"], 'rule': corpus_json[i]["name"]})

    # Compute embeddings
    corpus_embeddings = embedder.encode(corpus, convert_to_tensor=True)

    # save corpus_embeddings to pickle file
    with open('embeddings.pkl', "wb") as fOut:
        pickle.dump({'sentences': corpus, 'embeddings': corpus_embeddings}, fOut, protocol=pickle.HIGHEST_PROTOCOL)

# Query sentences:

# Find the closest 5 sentences of the corpus for each query sentence based on cosine similarity
top_k = min(5, len(corpus))
def search(query):
    query_embedding = embedder.encode(query, convert_to_tensor=True)

    # We use cosine-similarity and torch.topk to find the highest 5 scores
    hits = util.semantic_search(query_embedding, corpus_embeddings, top_k=5)
    # hits contains dicts of (corpus_id, score)
    # convert corpus_ids to sentences
    hits = [{'text': corpus[hit['corpus_id']]['rule'], 'score': hit['score']} for hit in hits[0]]
    return hits;
# use flask to create an api
app = Flask(__name__)

@app.route('/search', methods=['POST'])
def search_api():
    data = request.get_json()
    query = data['query']
    res = search(query)
    print(res)
    return jsonify({'status': 'success', 'data': list(res)})

if __name__ == '__main__':
    app.run(host='localhost', port=5000, debug=True)
