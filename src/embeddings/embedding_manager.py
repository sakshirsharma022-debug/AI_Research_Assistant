from sentence_transformers import SentenceTransformer

class EmbeddingManager:

    def __init__(self, model_name="all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)


    def generate_embeddings(self, chunks):

        texts = [chunk.page_content for chunk in chunks]

        embeddings = self.model.encode(texts)

        return embeddings

