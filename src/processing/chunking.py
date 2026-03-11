from langchain_text_splitters import RecursiveCharacterTextSplitter

class ChunkProcessor:
    def __init__(self, chunk_size=500, chunk_overlap=100):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size = chunk_size,
            chunk_overlap = chunk_overlap
        )

    def split_documents(self, documents):

        chunks = self.text_splitter.split_documents(documents)

        return chunks