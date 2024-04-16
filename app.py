from flask import Flask, request, jsonify
from core.pipeline import SentenceTransformerPineconePipeline
import json

app = Flask(__name__)
SENTENCETRANSFORMER_PINECONE_PIPELINE = SentenceTransformerPineconePipeline()



@app.route('/api/95buddha/upsert', methods=['POST'])
def api_upsert_to_pinecone():
    """Up một chunk lên pinecone cloud"""
   
    try:
        data = request.form
        uid = SENTENCETRANSFORMER_PINECONE_PIPELINE.encoding_and_upsert(
            input={
                "text": data["text"],
                "metadata": {
                    "authors": [i.strip() for i in data["authors"].split(",")]
                }
            }
        )    
        if uid == -1:
            return jsonify({'message': "duplicated, let's skipping"}), 200
        else:
            return jsonify({'message': f"upserted {uid}"}), 200
    except Exception as e:
        return jsonify({'error': f'No params provided {e}'}), 400


@app.route('/api/95buddha/query', methods=['POST'])
def search():
    try:
        data = request.form
        r = SENTENCETRANSFORMER_PINECONE_PIPELINE.encoding_and_query(
            query=str(data["query"]), 
            top_k=int(data["top_k"])
        ).to_dict()        
        
        return jsonify({"matches": r["matches"]}), 200
    
    except Exception as e:
        return jsonify({'error': str(e)}), 400




if __name__ == '__main__':
    app.run(debug=True, port=3322)