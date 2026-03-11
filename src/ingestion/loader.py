from langchain_community.document_loaders import PyPDFLoader
from pathlib import Path
import os



class PDFLoader:
    
    
    def __init__(self, pdf_directory: str):
        """  
        Args:
            pdf_directory: Path where PDFs are stored
        """

        self.pdf_directory = pdf_directory


    def load_pdfs(self):
            documents = []

            for file in os.listdir(self.pdf_directory):

                if file.endswith(".pdf"):
                    file_path = os.path.join(self.pdf_directory, file)

                    loader = PyPDFLoader(file_path)
                    pages = loader.load()

                    documents.extend(pages)

                return documents
        

