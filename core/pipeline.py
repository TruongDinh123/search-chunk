from core.modules.embedder import SentenceEmbedder
from core.modules.pinecone import CustomPinecone
from core.modules.utils import generate_unique_id, get_passage_emb_template, get_query_emb_template



class SentenceTransformerPineconePipeline:
    
    def __init__(self, sentence_transformer_model="intfloat/e5-small"):
        self.e = SentenceEmbedder(sentence_transformer_model)
        self.p = CustomPinecone()
        self.sentence_transformer_model = sentence_transformer_model
    
    def get_pipeline_profile(self):
        return f"{self.sentence_transformer_model}"
        
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
        r = self.encoding_and_query(
            query=get_passage_emb_template(
                title=input["metadata"]["title"],
                types=", ".join(input["metadata"]["types"]),
                authors=", ".join(input["metadata"]["authors"]),
                chunk_text=input["text"]            
            )
        )
        if ( r["matches"] == [] or r["matches"][0]["score"] < 0.95):
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
        v = self.e.encode_chunk(get_query_emb_template(query))
        return self.p.search(vector=v, top_k=top_k)
    
        
        
        

        