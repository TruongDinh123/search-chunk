from core.pipeline import SentenceTransformerPineconePipeline
from core.modules.pinecone import CustomPinecone
from core.modules.utils import get_passage_emb_template, get_query_emb_template


# p = CustomPinecone()
# p.delete_all_vector(namespace="ns1")


# Insert data
pipeline = SentenceTransformerPineconePipeline()
# text = input("CHUNK_TEXT: ")
# types = input("TYPE: ")
# title = input("TITLE: ")
# authors = input("AUTHORS: ")
# text = "Nếu bạn chinh phục được chính mình, bạn có thể chinh phục được cả thế giới."
# types = "Tiểu thuyết"
# title = "Nhà Giả Kim"
# authors = "Paulo Coelho"

# print(
#     pipeline.encoding_and_upsert(
#         input={
#             "text": text,
#             "metadata": {
#                 "document_id": "this-is-test-document-id",
#                 "title": title,
#                 "types": [t.strip() for t in types.split(",")],                    
#                 "authors": [author.strip() for author in authors.split(",")],
#                 "pipeline_profile": pipeline.get_pipeline_profile()
#             }
#         }
#     )
# )
# Query data
import json
from core.modules.bm25 import CustomBM25
pipeline = SentenceTransformerPineconePipeline("intfloat/e5-small")
while True:
    q = input("[>] ")
    r = pipeline.encoding_and_query(query=q, top_k=10)
    print(r)

    








