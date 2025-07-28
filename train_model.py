import gradio as gr
from datetime import datetime
import json
from src.tools.mongo_client import MongoDBClient

mongo_client = MongoDBClient()
users = mongo_client.get_all_user_ids("chat_collection")

PAGE_SIZE = 10

# Simulate MongoDB text list (if you want to test locally)
# full_results = [f"Sample text #{i}" for i in range(1, 101)]  # 100 samples

def query_by_time(user_id, start_date, end_date):
    if user_id is not None:
        docs = mongo_client.find_documents("chat_collection", {"user_id": user_id})
    elif start_date and end_date:
        start = datetime.fromtimestamp(start_date)
        end = datetime.fromtimestamp(end_date)
        docs = mongo_client.get_user_docs_in_time("chat_collection", start, end)
    else:
        return ["Please select a user ID or a time range"]

    return docs

def get_page_items(results, page):
    start = page * PAGE_SIZE
    end = start + PAGE_SIZE
    return results[start:end]

def display_page(results, page):
    items = get_page_items(results, page)
    text_updates = []
    for i in range(PAGE_SIZE):
        if i < len(items):
            text_updates.extend([gr.update(visible=True), gr.update(value=items[i])])
        else:
            text_updates.extend([gr.update(visible=False), gr.update(value="")])
    return text_updates

with gr.Blocks() as demo:
    # UI: filters
    with gr.Row():
        user_id = gr.Dropdown(choices=[None] + users, label="User ID", value=None)
        start_date = gr.DateTime(label="Start Date")
        end_date = gr.DateTime(label="End Date")
    submit_btn = gr.Button("Submit")

    # State variables
    full_results = gr.State([])
    page_number = gr.State(0)

    # Output area
    textboxes = []
    with gr.Column() as output_column:
        for i in range(PAGE_SIZE):
            with gr.Row(visible=False) as row:
                txt = gr.Textbox(label=f"Text {i+1}", interactive=False, lines=2, scale=5)
                btn = gr.Button("Select", scale=1)
                btn.click(fn=lambda text=txt: print(f"[Selected]: {text}"), inputs=[], outputs=[])
                textboxes.append((row, txt))

    with gr.Row():
        prev_btn = gr.Button("Previous")
        next_btn = gr.Button("Next")

    def on_submit(user_id, start_date, end_date):
        results = query_by_time(user_id, start_date, end_date)
        page = 0
        updates = display_page(results, page)
        return [results, page] + updates

    submit_btn.click(
        fn=on_submit,
        inputs=[user_id, start_date, end_date],
        outputs=[full_results, page_number] + [r for r, t in textboxes for r in (r, t)]
    )

    def go_page(direction, results, page):
        if not results:
            return [page] + [gr.update() for _ in range(PAGE_SIZE * 2)]
        new_page = max(0, page + direction)
        if new_page * PAGE_SIZE >= len(results):
            new_page = page  # stay on current page if over
        updates = display_page(results, new_page)
        return [new_page] + updates

    next_btn.click(
        fn=lambda results, page: go_page(1, results, page),
        inputs=[full_results, page_number],
        outputs=[page_number] + [r for r, t in textboxes for r in (r, t)]
    )

    prev_btn.click(
        fn=lambda results, page: go_page(-1, results, page),
        inputs=[full_results, page_number],
        outputs=[page_number] + [r for r, t in textboxes for r in (r, t)]
    )

demo.launch()
