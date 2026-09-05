from fastapi import FastAPI
import gradio as gr
import sys
import os

# Ensure the root directory is on the path so we can import app.py
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app import demo

app = FastAPI()

# Mount the Gradio app to the FastAPI app for Vercel
app = gr.mount_gradio_app(app, demo, path="/")
