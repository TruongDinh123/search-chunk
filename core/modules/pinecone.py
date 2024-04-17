from dotenv import dotenv_values
from pinecone import Pinecone, ServerlessSpec



config = dotenv_values(".env")

class CustomPinecone():

    def __init__(self):    

        pc = Pinecone(
            api_key=config["PINECONE_API_KEY"]
        )
        
        self.index = pc.Index(config["PINECONE_INDEX_NAME"])


    def upsert_chunk(self, vector):
        
        try:
                        
            self.index.upsert(
                vectors=[
                    {
                    "id": vector["id"], 
                    "values": vector["values"], 
                    "metadata": vector["metadata"]
                    } 
                ],
                namespace= "ns1"
            )
            
        except Exception as e:
            print(e)


    def upsert_chunks(self, vectors):
        try:
            
            self.index.upsert(
                vectors=vectors
            )
            
        except Exception as e:
            print(e)
    
    
    def delete_vector(self, ids: list[str], namespace):
        self.index.delete(ids=ids, namespace=namespace)


    def delete_all_vector(self, namespace):
        self.index.delete(
            delete_all=True, 
            namespace=namespace
        )
        print("Deleted successfully")
        
    def search(self, vector, top_k = 3):
        return self.index.query(
            namespace="ns1",
            vector=vector,
            top_k=top_k,
            # include_values=True,
            include_metadata=True
        )
    
    def search_with_document_id(self, document_id):
        return self.index.query(
            filter={
                "document_id": document_id
            },
            include_metadata=True
        )
        