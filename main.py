import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
import gradio as gr

textbox = gr.Textbox()
dropdown = gr.Dropdown()

gr.load_chat("http://localhost:11434/v1/", model="llama3.2", token="***").launch()