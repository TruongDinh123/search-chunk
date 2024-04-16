from core.modules.embedder import SentenceEmbedder
from core.modules.pinecone import CustomPinecone
from core.modules.utils import generate_unique_id




class SentenceTransformerPineconePipeline:
    
    def __init__(self):
        self.e = SentenceEmbedder("sentence-transformers/all-MiniLM-L6-v2")
        self.p = CustomPinecone()
        
        
    def encoding_and_upsert(self, input):
        """
        Run sentencetransformer + pinecone pipeline 

        Args:
            input: {
                "text": "This is textttt" 
                "metadata": {
                    "author": ...
                    ...
                }
            }
        """
        
        v = self.e.encode_chunk(input["text"])
        
        # Kiểm tra xem có duplicate không
        r = self.encoding_and_query(query=input["text"])
        if (r["matches"][0]["score"] < 0.95):
            uid  = str(generate_unique_id())
            temp = {
                    "id": uid,
                    "values": v,
                    "metadata": input["metadata"]
            }
            temp["metadata"]["text"] = input["text"] 
            self.p.upsert_chunk(temp)
            return uid
        else:
            print("Duplicated chunk, let's skip!")
            return -1
        
    def encoding_and_query(self, query, top_k=3):
        v = self.e.encode_chunk(query)
        return self.p.search(vector=v, top_k=top_k)
    
        
        
        

        