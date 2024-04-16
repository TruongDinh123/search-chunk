# from core.modules.embedder import SentenceEmbedder
# from core.modules.pinecone import CustomPinecone
# from core.modules.utils import generate_unique_id



# e = SentenceEmbedder("sentence-transformers/all-MiniLM-L6-v2")
# # while True:

# # q = input("[Query] ")
# q = "Mỗi người dường như đều biết rất rõ ràng về cách người khác nên sống ra sao, nhưng lại không biết gì về chính con đường của mình"
# v = e.encode_chunk(q)
# pc = CustomPinecone()
# pc.upsert_chunk(
#     {
#         "id": generate_unique_id(),
#         "values": v[0],
#         "metadata": {
#             "doc": "Nhà giả kim",
#             "author": "Paulo Coelho"
#         }
#     }
# )
# print(v)
# print("----------------" * 5)


from core.pipeline import SentenceTransformerPineconePipeline
from core.modules.pinecone import CustomPinecone



# p = CustomPinecone()
# p.delete_all_vector(namespace="ns1")



# Insert data
pipeline = SentenceTransformerPineconePipeline()
while True:    
    
    input_text = input("q: ")
    pipeline.encoding_and_upsert(
        input={
            "text": input_text,
            "metadata": {
                "authors": ["Paullo Coelho"],
            }
        }
    )

# Query data
import json
pipeline = SentenceTransformerPineconePipeline()
r = pipeline.encoding_and_query(query="Muốn yêu thương người khác")
print(type(r.to_dict()))

# print([i["score"] for i in r["matches"]])


