import gradio as gr
from rag import retrieve_generate
from create_vector_db import update_db
from langchain_core.messages import HumanMessage, AIMessage

def parse_context(context):
    metadata = list(filter(lambda x: x.startswith("["), context.split("\n")))
    concat = []
    for i in range(len(metadata)):
        if i > 0 and i%2==1:
            concat.append(metadata[i-1].strip("[]") + " " + metadata[i].strip("[]"))
    return "".join(line + "\n" for line in concat)


def convert_history(history):
    messages = []

    for message in history:
        if message["role"] == "user":
            messages.append(HumanMessage(content=message["content"]))
        elif message["role"] == "assistant":
            messages.append(AIMessage(content=message["content"]))

    return messages


def chat(message, history, university):

    history = history or []
    langchain_history = convert_history(history)

    response, context = retrieve_generate(
        message,
        university,
        langchain_history
    )

    parsed = parse_context(context)

    history.append({
        "role": "user",
        "content": message
    })

    history.append({
        "role": "assistant",
        "content": response
    })

    return history, parsed


def upload_file(file):
    if file is None:
        return "No file selected."

    update_db(file, key=dropdown.value)
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

        upload_button = gr.Button("Dosya Ekle")

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