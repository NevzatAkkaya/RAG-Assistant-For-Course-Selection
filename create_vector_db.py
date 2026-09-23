from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaEmbeddings

import os

splitter = RecursiveCharacterTextSplitter(
    chunk_size=600,
    chunk_overlap=70,
    separators=["\n\n", "\n", ".", " ", ""]
)

embeddings = OllamaEmbeddings(model="bge-m3")

def rand_key(key):
    keys = os.listdir("db")
    if key + str(len(key)) in keys:
        return rand_key(key+"1")
    else:
        return key + str(len(key))

def parse_split(document):
    if document.endswith(".pdf"):
            loader = PyPDFLoader(document)
            documents = loader.load()
            for doc in documents:
                doc.metadata["source"] = os.path.basename(document)
            chunks = splitter.split_documents(documents)
    else:
         with open(document, "r") as doc:
            parsed = "".join(row for row in doc)
            documents = Document(page_content=parsed, metadata={"source": os.path.basename(document)})
            chunks = splitter.split_documents([documents])  

    return chunks          
       

def create_db(document, key="default"):
    keys = os.listdir("db")
    chunks = parse_split(document)
    vector_store = FAISS.from_documents(documents=chunks, embedding=embeddings)
    if key in keys:
        vector_store.save_local(os.path.join("db",rand_key(key)))
    else:
        vector_store.save_local(os.path.join("db",key))


def update_db(document, key="default"):
    chunks = parse_split(document)
    if os.path.exists(os.path.join("db",key,"index.faiss")):
        vector_store = FAISS.load_local(os.path.join("db",key), embeddings=embeddings, allow_dangerous_deserialization=True)
        vector_store.add_documents(documents=chunks)
        vector_store.save_local(os.path.join("db",key))
    else:
        create_db(document, key)
