import gradio as gr
from backend import handle_evaluation
import json

test_statements = ['What causes failure of the oil engine?', 'What causes failure of the electric engine?']

# Function to add a message to the chatbot history
def add_message(history, message, use_ontology):
    history.append((message, None))
    return history, "", use_ontology

# Function to generate the initial bot response (only show statements)
def generate_prompt_and_bot_response(history, user_input, use_ontology, ontology_file):
    history.append((user_input, None))  # Add user input to history
    
    # Extract the file path if the file is uploaded, otherwise use None
    ontology_path = ontology_file if ontology_file is not None else None
    
    # Call the handle_evaluation function to get the response
    response = handle_evaluation(user_input, use_ontology, ontology_path)
    response_data = json.loads(response)
    
    # Only display the statements initially
    statements = response_data.get("statements", "")
    history[-1] = (f"{user_input} (Ontology: {'Enabled' if use_ontology else 'Disabled'})", statements)
    
    # Store the response data (including details) in state
    return history, response_data

# Function to show details when the "Details" button is clicked
def show_details(history, response_data):
    # Get the details from the stored response
    details = response_data.get("details", "")
    if details:
        history.append(("Details:", details))
    return history

# Function to update the textbox when an example is selected
def update_textbox(selected_statement):
    return selected_statement

with gr.Blocks(title="Engine Failure Evaluation") as demo:
    gr.Markdown("## Evaluate Statements About Engine Failures")

    with gr.Row():
        chat_input = gr.Textbox(placeholder="Enter statement or select an example...", lines=3, label="Input", show_label=True)
        ontology_checkbox = gr.Checkbox(label="Use Ontology", value=False, interactive=True)
        submit_button = gr.Button("Submit")

    with gr.Row():
        # Changed `type` from "file" to "filepath"
        ontology_upload = gr.File(label="Upload Custom Ontology (Optional)", type="filepath")

    chatbot = gr.Chatbot(label="Chatbot")

    # State to store the response data for later use
    response_state = gr.State()

    # Event triggers
    submit_button.click(add_message, [chatbot, chat_input, ontology_checkbox], [chatbot, chat_input, ontology_checkbox])
    submit_button.click(generate_prompt_and_bot_response, [chatbot, chat_input, ontology_checkbox, ontology_upload], [chatbot, response_state])

    # Button to display additional details (only relevant if ontology is used)
    details_button = gr.Button("Show Details")
    details_button.click(show_details, [chatbot, response_state], chatbot)

    chat_input.submit(generate_prompt_and_bot_response, [chatbot, chat_input, ontology_checkbox, ontology_upload], [chatbot, response_state])

    # Example Statements Dropdown
    with gr.Row():
        gr.Markdown("### Example Statements:")
    example_dropdown = gr.Dropdown(choices=test_statements, label="Select an example statement", interactive=True)
    example_dropdown.change(update_textbox, example_dropdown, chat_input)

demo.launch(debug=True, share=False, server_name="0.0.0.0")

