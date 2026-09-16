from langchain_text_splitters import RecursiveCharacterTextSplitter, HTMLHeaderTextSplitter
import pandas as pd
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader

splitter = RecursiveCharacterTextSplitter(
    chunk_size=600,
    chunk_overlap=70,
    separators=["\n\n", "\n", ".", " ", ""]
)

embeddings = GoogleGenerativeAIEmbeddings(model = "gemini-embedding-001")

def parse_split(document):
    if document.endswith(".pdf"):
            loader = PyPDFLoader(document)
            documents = loader.load()
            chunks = splitter.split_documents(documents)
    else:
         with open(document, "r") as doc:
            parsed = "".join(row for row in doc)
            documents = Document(page_content=parsed)
            chunks = splitter.split_documents([documents])  

    return chunks          

         

def create_db(document):
    chunks = parse_split(document)
    pass

def update_db(document):
    chunks = parse_split(document)
    pass