# GenAI RAG QA Bot

This project is a Retrieval-Augmented Generation (RAG) application built using LangChain, Gradio, and ChromaDB. It functions as a document QA bot that allows users to upload PDF files and ask questions about their content using a custom IBM WatsonX Large Language Model integration.

![QA Bot Interface](QA_bot.png)

## Prerequisites

Before setting up the project, ensure you have the following installed on your machine:
* Python 3.9
* Git

## Step-by-Step Setup Guide

### Step 1: Clone the Repository
Open your terminal and clone the repository to your local machine.
```bash
git clone https://github.com/WilWilbert123/GenAI-RAG-LangChain.git
cd GenAI-RAG-LangChain
```

### Step 2: Set up a Virtual Environment
It is highly recommended to isolate the project dependencies using a virtual environment.
```bash
python -m venv venv
source venv/bin/activate
```

### Step 3: Install Required Dependencies
Install all the necessary Python packages using the provided requirements file.
```bash
pip install -r requirements.txt
```

### Step 4: Configure Environment Variables
Create a file named `.env` in the root directory of the project. This file is required to authenticate with the IBM WatsonX API. Add the following lines to the file, replacing the placeholder values with your actual credentials:
```env
WATSONX_API_KEY=your_api_key_here
WATSONX_PROJECT_ID=your_project_id_here
WATSONX_URL=your_url_here
```

### Step 5: Start the Application
Run the main application script to start the Gradio web server.
```bash
python app.py
```

### Step 6: Access the Web Interface
Once the server starts, it will display a local URL in the terminal (typically http://127.0.0.1:7860 or similar). Open this link in your web browser.

## Usage Guide

1. **Upload Document**: Click on the file upload area labeled "Select PDF File" and choose a PDF from your computer.
2. **Analyze**: Click the "Analyze Document" button. The system will extract the text, chunk it, and store the embeddings in a local Chroma vector database.
3. **Wait for Status**: Look at the Status box to confirm that the document was processed successfully.
4. **Ask Questions**: Type your query in the "Your Question" box.
5. **Get Answer**: Click the "Get Answer" button. The system will retrieve the most relevant chunks of text from your document and use the IBM WatsonX LLM to generate an accurate summary or answer.

## Architecture and Tools

* **LangChain**: Orchestrates the RAG pipeline.
* **PyPDFLoader**: Extracts text from the uploaded PDF documents.
* **RecursiveCharacterTextSplitter**: Splits the document into optimized chunks for embedding.
* **Chroma**: Local vector database for storing and querying text embeddings.
* **HuggingFaceEmbeddings**: Generates the vector representations of the text.
* **IBM WatsonX REST API**: Custom integration for LLM inference (using Llama 3 70B).
* **Gradio**: Provides the minimalist, dark-mode web user interface.
