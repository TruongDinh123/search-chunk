from rank_bm25 import BM25Okapi
from core.modules.utils import get_passage_emb_template



class CustomBM25():
    
    
    def rerank(self, query, corpus, top_k=3):
        query = f"{query} {query}"
        corpus_text = [
            
            get_passage_emb_template(
                title       = item["metadata"]["title"],
                authors     = " ".join(item["metadata"]["authors"]),
                types       = " ".join(item["metadata"]["types"]),
                chunk_text  = item["metadata"]["text"]
            )
            
            for item in corpus
        ]
        tokenized_query = query.split(" ")
        tokenized_corpus = [doc.split(" ") for doc in corpus_text]
        bm25 = BM25Okapi(tokenized_corpus)
        doc_scores = bm25.get_scores(tokenized_query)
        # Sắp xếp lại các chỉ số của corpus dựa trên xếp hạng
        ranked_indices = sorted(range(len(doc_scores)), key=lambda i: doc_scores[i], reverse=True)
        sorted_corpus = [corpus[i] for i in ranked_indices]
        return sorted_corpus
