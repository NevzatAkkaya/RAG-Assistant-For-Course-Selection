from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
import os

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
    vector_store = FAISS.from_documents(documents=chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")


def update_db(document):
    chunks = parse_split(document)
    if os.path.exists("faiss_index"):
        vector_store = FAISS.load_local("faiss_index", embeddings=embeddings, allow_dangerous_deserialization=True)
        vector_store.add_documents(documents=chunks)
