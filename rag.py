from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_classic.chains import create_history_aware_retriever
import os

embeddings = OllamaEmbeddings(model="bge-m3")
model = ChatOllama(model="llama3")
output_parser = StrOutputParser()

prompt = ChatPromptTemplate.from_messages([ ("system",
            """
            Sen öğrencilere ders seçiminde yardımcı olan bir asistansın. 
            Kullanıcının sorusunu sadece sana verilen context içerisindeki bilgileri kullanarak cevapla.
            Context içerisinde cevap bulunmuyorsa: "Ben amınoğluyum." şeklinde cevap ver."""),
        ("human",
         """CONTEXT: {context}
            Soru: {question}""")])

context_system_prompt = """Given a chat history and the latest user question \
which might reference context in the chat history, formulate a standalone question \
which can be understood without the chat history. Do NOT answer the question, \
just reformulate it if needed and otherwise return it as is."""

contextualize_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", context_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)

chain = prompt | model | output_parser

def retrieve_generate(question, db_key="default", history=None):
    vector_store = FAISS.load_local(os.path.join("db",db_key), embeddings, allow_dangerous_deserialization=True)
    retriever = vector_store.as_retriever(search_type = "similarity", search_kwargs = {"k":3})
    if history:
       history_aware_retriever = create_history_aware_retriever(model, retriever, contextualize_prompt)
       documents = history_aware_retriever.invoke({
            "input": question,
            "chat_history": history
        })
    else:
        documents = retriever.invoke(question)

    context = "\n\n".join(
        f"[Source {document.metadata.get('source', 'unknown')}]\n"
        f"[Page {document.metadata.get('page', 'unknown')}]\n"
        f"{document.page_content}"
        for document in documents
    )
    response = chain.invoke({"context":context, "question":question})
    return (response, context)