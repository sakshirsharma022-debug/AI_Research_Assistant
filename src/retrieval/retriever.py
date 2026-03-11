class Retriever:

    def __init__(self, vector_store, embedding_model):
        self.vector_store = vector_store
        self.embedding_model = embedding_model

    def retrieve(self, query, k=5):
        query_embedding = self.embedding_model.encode(query)

        results = self.vector_store.search(query_embedding, k)

        return results