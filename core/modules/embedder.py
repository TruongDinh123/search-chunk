from sentence_transformers import SentenceTransformer, CrossEncoder, util
import pinecone
import uuid


# Models
# "sentence-transformers/all-MiniLM-L6-v2"
# "sentence-transformers//models/stsb-xlm-r-multilingual"


class SentenceEmbedder():
    
    def __init__(self, model_name):
        self.encoder = SentenceTransformer(model_name)
        
    def encode_chunk(self, chunk: str):
        vector = self.encoder.encode(
            [chunk], 
            convert_to_tensor=True, 
            show_progress_bar=False
        )[0]
        return vector.tolist()

    def encode_chunks(self, chunks: list[str]):
        vectors = self.encoder.encode(
            chunks, 
            convert_to_tensor=True, 
            show_progress_bar=False
        )
        return vectors
        


