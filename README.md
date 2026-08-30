# GenAI-RAG-LangChain

This repository contains a Retrieval-Augmented Generation (RAG) application built using LangChain, Gradio, and Chroma. The application allows users to upload PDF documents and ask questions about the content of those documents in real time.

## Project Overview

This project was built as part of the "Generative AI Applications with RAG and LangChain" course. It serves as an AI assistant for processing and extracting insights from documents to optimize research productivity and decision-making.

## Prerequisites

Before running this application, ensure you have the following installed:
- Python 3.9 or higher
- pip (Python package installer)

## Step-by-Step Installation Guide

### Step 1: Clone the Repository
Clone this repository to your local machine using git.
```bash
git clone https://github.com/WilWilbert123/GenAI-RAG-LangChain.git
cd GenAI-RAG-LangChain
```

### Step 2: Create a Virtual Environment
It is highly recommended to use a virtual environment to manage project dependencies.
```bash
python3 -m venv venv
source venv/bin/activate
```

### Step 3: Install Dependencies
Install all required Python packages using the provided requirements file.
```bash
pip install -r requirements.txt
```

### Step 4: Environment Variables
Create a `.env` file in the root directory if you plan to use IBM Watsonx credentials. 
```env
WATSONX_API_KEY=your_api_key_here
WATSONX_PROJECT_ID=your_project_id_here
WATSONX_URL=your_url_here
```
Note: If these credentials are not provided, the application will automatically fall back to using local HuggingFace embeddings (`all-MiniLM-L6-v2`) and a mock LLM for testing purposes.

### Step 5: Run the Application
Start the Gradio web server.
```bash
python3 app.py
```

### Step 6: Access the Web Interface
Once the server is running, open your web browser and navigate to the provided local URL (typically http://127.0.0.1:7860). 

## How to Use the Application

1. Open the application in your web browser.
2. Click on the "Upload PDF Document" component to upload a research document or certificate.
3. Wait for the "Status" box to confirm that the document was processed successfully.
4. Type your question in the "Ask a Question about the Document" text box.
5. Click the "Ask QA Bot" button.
6. The bot's answer will appear in the "QA Bot Answer" box below.

## Architecture and Tools Used

- LangChain: Core framework for orchestrating the RAG pipeline.
- PyPDFLoader: Extracts text from the uploaded PDF files.
- RecursiveCharacterTextSplitter: Splits the extracted text into manageable chunks.
- HuggingFaceEmbeddings: Converts text chunks into vector representations (fallback for Watsonx).
- Chroma: Vector database for storing and retrieving the embeddings.
- Gradio: Framework used to build the interactive web user interface.
