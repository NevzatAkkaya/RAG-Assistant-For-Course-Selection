from langchain_text_splitters import RecursiveCharacterTextSplitter
import pandas as pd
from pypdf import PdfReader
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS

splitter = RecursiveCharacterTextSplitter(
    chunk_size=80,
    chunk_overlap=20,
    separators=["\n\n", "\n", ".", " ", ""]
)

embeddings = GoogleGenerativeAIEmbeddings(model = "gemini-embedding-001")

def parse(document):
    if document.endswith(".pdf"):
            reader = PdfReader(document)
            parsed = ""
            for num in range(len(reader.pages)):
                page = reader.get_page(num)
                if page.extract_text() is not None:
                    parsed = parsed + page.extract_text()
    else:
         with open(document, "r") as doc:
            parsed = "".join(row for row in doc)            

         

def create_db(text):        
    chunks = splitter.split_text(text)
    pass

def update_db(text):         
    chunks = splitter.split_text(text)
    pass