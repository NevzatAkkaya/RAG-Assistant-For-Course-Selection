import gradio as gr

from rag import retrieve_generate
from create_vector_db import update_db


def chat(message, history, university):
    response, context = retrieve_generate(message, university, history)
    parsed = parse_context(context)
    return response, parsed

def parse_context(context):
    return "".join((str(line[0]) + str(line[1])).strip("[]") + "\n" for line in list(map(lambda y: y.split(" ")[:2],filter(lambda x: True if x.startswith("[Page") and len(x.split(" ")) < 3 else False, context.split("\n")))))

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

        upload_button = gr.Button("Transkript Ekle")

    upload_status = gr.Textbox(
        label="Status",
        interactive=False
    )

    source_text = gr.Textbox(
        label="Source Files",
        interactive=False
    )

    dropdown = gr.Dropdown(
    choices=["BOUN", "ITU", "IU", "METU"],
    label="University",
    value="BOUN"
    )  

    msg.submit(
        chat,
        inputs=[msg, chatbot, dropdown],
        outputs=[chatbot, source_text]
    )

    upload_button.click(
        upload_file,
        inputs=file,
        outputs=upload_status
    )



demo.launch(share=False)