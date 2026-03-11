import faiss
import numpy as np

class FAISSVectorStore:

    def __init__(self, embedding_dim):

        self.embedding_dim = embedding_dim

        self.index = faiss.IndexFlatL2(embedding_dim)

        self.text_chunks = []

    def add_embeddings(self, embeddings, chunks):

        embeddings = np.array(embeddings).astype("float32")

        self.index.add(embeddings)

        self.text_chunks.extend(chunks)

    def search(self, query_embedding, k=5):

        query_embedding = np.array([query_embedding]).astype("float32")

        distances, indices = self.index.search(query_embedding, k)

        results = [self.text_chunks[i] for i in indices[0]]

        return results