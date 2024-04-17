from flask import Flask, request, jsonify
from core.pipeline import SentenceTransformerPineconePipeline
from core.modules.pinecone import CustomPinecone

import json

app = Flask(__name__)
SENTENCETRANSFORMER_PINECONE_PIPELINE = SentenceTransformerPineconePipeline()



@app.route('/api/95buddha/chunk/upsert', methods=['POST'])
def api_upsert_to_pinecone():
    """Up một chunk lên pinecone cloud"""
   
    try:
        data = request.form
        uid = SENTENCETRANSFORMER_PINECONE_PIPELINE.encoding_and_upsert(
            input={
                "text": data["text"],
                "metadata": {
                    "document_id": data["document_id"],
                    "title": data["title"],
                    "types": [t.strip() for t in data["types"].split(",")],                    
                    "authors": [author.strip() for author in data["authors"].split(",")],
                    "pipeline_profile": SENTENCETRANSFORMER_PINECONE_PIPELINE.get_pipeline_profile()
                }
            }
        )
            
        if uid == -1:
            return jsonify({'message': "duplicated, let's skipping"}), 200
        else:
            return jsonify({'message': f"upserted {uid}"}), 200
    except Exception as e:
        return jsonify({'error': f'No params provided {e}'}), 400


@app.route('/api/95buddha/chunk/query', methods=['POST'])
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


@app.route('/api/95buddha/chunk/delete', methods=['POST'])
def delete_vector():
    try:
        data = request.form
        pc = CustomPinecone()
        
        pc.delete_vector(ids=[data["id"]], namespace="ns1")
        return jsonify({"message": f"Deleted successfully {data['id']}"}), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 400
    
    

if __name__ == '__main__':
    app.run(debug=True, port=3322)