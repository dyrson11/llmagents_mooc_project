# llmagents_mooc_project
This project implements a multi-agent system for machine translation, focusing on low-resource languages of the Americas. It includes agents for dictionary lookup, translation, evaluation, and prompt rewriting, orchestrated through a graph-based workflow.

## Features
- Dictionary-based translation assistance
- Translation finalization and evaluation
- Prompt rewriting for agent improvement
- Support for low-resource languages (e.g., Aymara)
- Library Dependencies

## Requirements
Install the following Python libraries:

- pandas
- tqdm
- python-dotenv
- sacrebleu
- torch
- transformers
- sentence-transformers
- langchain
- langchain_huggingface
- langgraph

You can install them with:
`pip install pandas tqdm python-dotenv sacrebleu torch transformers sentence-transformers langchain langchain_huggingface langgraph`

## Instructions for Running
Set up environment variables
Create a .env file with the HF API key (HF_TOKEN).

## Run the main script
From the project root, execute:
`python main.py`

Unfortunately, the command is not parameterized yet. However, you can adjust parameters such as source/target language, mode, and data paths by editing main.py.

## Output
Translations and evaluation metrics are saved in the translations directory.
Agent messages are saved in the output/messages/ directory.
