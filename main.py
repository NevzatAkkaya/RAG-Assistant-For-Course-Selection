import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

google_api_key = os.getenv("API_KEY")

model = ChatGoogleGenerativeAI(
    model = "gemini-2.5-flash-lite"
)

prompt = ChatPromptTemplate(
    [
        ("system", "You are an assistant who is tasked to explain subjets upon asked"),
        ("human", """" Subject: {topic} """)

    ]
)

user_prompt = "LLM nedir ?"

response = model.invoke(user_prompt)

print(response.content)