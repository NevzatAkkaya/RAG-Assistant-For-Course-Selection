import gradio as gr

from rag import retrieve_generate
from create_vector_db import update_db


def chat(message, history):
    return retrieve_generate(message)


def upload_file(file):
    if file is None:
        return "No file selected."

    update_db(file)
    return f"Added {file}"


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

        upload_button = gr.Button("Add to knowledge base")

    upload_status = gr.Textbox(
        label="Status",
        interactive=False
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