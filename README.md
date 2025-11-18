# Personal Finance Chatbot

This project is a simple Streamlit app that uses a Hugging Face text-generation pipeline (IBM Granite instruct model by default)
to provide personalized financial guidance, budget summaries, and spending insights.

## Quickstart (Windows)

1. Open PowerShell and navigate to the project folder:

2. (Optional) Create a virtual environment:

3. Install dependencies:

4. Run the app:

## Notes
- If the HF model is not available locally or network access is blocked, the app returns a fallback response (so the UI remains usable).
- To use a different model, edit `model_pipeline.py` and set `HF_MODEL_NAME` to your model identifier.
