import gradio as gr

from rag import retrieve_generate
from create_vector_db import update_db


def chat(message, history):
    response, context = retrieve_generate(message)
    return response


def upload_file(file):
    if file is None:
        return "No file selected."

    update_db(file)
    return f"Added {file}"

def select_school(val):
    return val

with gr.Blocks() as demo:

    gr.Markdown("# Course Selection RAG")

    chatbot = gr.Chatbot()

    msg = gr.Textbox(
        placeholder="Ask a question...",
        label="Question"
    )

    with gr.Row():
        file = gr.File(
            label="Upload document",
            file_types=[".pdf", ".txt"],
            type="filepath"
        )

        upload_button = gr.Button("Transkript Ekle")

    upload_status = gr.Textbox(
        label="Status",
        interactive=False
    )

    dropdown = gr.Dropdown(
    choices=["BOUN", "ITU", "IU", "METU"],
    label="University",
    value="BOUN"
    )  

    msg.submit(
        chat,
        inputs=[msg, chatbot],
        outputs=chatbot
    )

    upload_button.click(
        upload_file,
        inputs=file,
        outputs=upload_status
    )



demo.launch(share=False)