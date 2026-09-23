from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
import os

embeddings = OllamaEmbeddings(model="bge-m3")
model = ChatOllama(model="llama3.2")
output_parser = StrOutputParser()

prompt = ChatPromptTemplate.from_messages([ ("system",
            """
            Sen öğrencilere ders seçiminde yardımcı olan bir asistansın. 
            Kullanıcının sorusunu sadece sana verilen context içerisindeki bilgileri kullanarak cevapla.
            Context içerisinde cevap bulunmuyorsa: "Bu bilgi dokümanlarda bulunmuyor." şeklinde cevap ver."""),
        ("human",
         """CONTEXT: {context}
            Soru: {question}""")])

chain = prompt | model | output_parser

def retrieve_generate(question, db_key="default", history=None):
    vector_store = FAISS.load_local(os.path.join("db",db_key), embeddings, allow_dangerous_deserialization=True)
    retriever = vector_store.as_retriever(search_type = "similarity", search_kwargs = {"k":3})
    documents = retriever.invoke(question)
    context = "\n\n".join(
    f"[Source {document.metadata.get('source', 'unknown')}]\n"
    f"[Page {document.metadata.get('page', 'unknown')}]\n"
    f"{document.page_content}"
    for document in documents
    )
    response = chain.invoke({"context":context, "question":question})
    return (response, context)