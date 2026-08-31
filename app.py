import os
os.environ["HF_HOME"] = os.path.abspath("./hf_cache")
import gradio as gr
from dotenv import load_dotenv

# Langchain imports
import requests
from typing import Any, List, Mapping, Optional
from langchain.llms.base import LLM
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

# Load environment variables
load_dotenv()

class WatsonXRESTLLM(LLM):
    model_id: str
    url: str
    project_id: str
    apikey: str
    token: str = ""
    
    @property
    def _llm_type(self) -> str:
        return "watsonx_rest"
        
    def _get_token(self):
        headers = {
            "Content-Type": "application/x-www-form-urlencoded",
            "Accept": "application/json"
        }
        data = {
            "grant_type": "urn:ibm:params:oauth:grant-type:apikey",
            "apikey": self.apikey
        }
        response = requests.post("https://iam.cloud.ibm.com/identity/token", headers=headers, data=data)
        response.raise_for_status()
        self.token = response.json()["access_token"]
        
    def _call(self, prompt: str, stop: Optional[List[str]] = None, **kwargs: Any) -> str:
        if not self.token:
            self._get_token()
            
        # Strip trailing slash from URL if present
        base_url = self.url.rstrip('/')
        endpoint = f"{base_url}/ml/v1/text/generation?version=2023-05-29"
        headers = {
            "Accept": "application/json",
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.token}"
        }
        payload = {
            "input": prompt,
            "parameters": {
                "decoding_method": "greedy",
                "max_new_tokens": 400,
                "min_new_tokens": 1,
                "temperature": 0.5,
                "stop_sequences": stop if stop else []
            },
            "model_id": self.model_id,
            "project_id": self.project_id
        }
        
        response = requests.post(endpoint, headers=headers, json=payload)
        if response.status_code == 401:
            self._get_token()
            headers["Authorization"] = f"Bearer {self.token}"
            response = requests.post(endpoint, headers=headers, json=payload)
            
        response.raise_for_status()
        result = response.json()
        return result["results"][0]["generated_text"]

# We'll try to import watsonx, but provide a fallback if credentials are missing
try:
    from langchain_ibm import WatsonxLLM, WatsonxEmbeddings
    HAS_WATSONX = True
except ImportError:
    HAS_WATSONX = False

def watsonx_embedding():
    """
    Lab function to create the Watsonx embedding model.
    Will fallback to HuggingFace embeddings if IBM credentials are not set.
    """
    api_key = os.getenv("WATSONX_APIKEY", "")
    project_id = os.getenv("WATSONX_PROJECT_ID", "")
    url = os.getenv("WATSONX_URL", "")

    if HAS_WATSONX and api_key and project_id and url:
        print("Using IBM Watsonx Embeddings...")
        return WatsonxEmbeddings(
            model_id="ibm/slate-125m-english-rtrvr",
            url=url,
            project_id=project_id,
            apikey=api_key
        )
    else:
        print("Using local HuggingFace Embeddings (Fallback)...")
        from langchain_community.embeddings import HuggingFaceEmbeddings
        return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

def get_llm():
    """
    Lab function to instantiate the LLM.
    """
    api_key = os.getenv("WATSONX_APIKEY", "")
    project_id = os.getenv("WATSONX_PROJECT_ID", "")
    url = os.getenv("WATSONX_URL", "")

    if HAS_WATSONX and api_key and project_id and url:
        print("Using IBM Watsonx LLM...")
        parameters = {
            "decoding_method": "greedy",
            "max_new_tokens": 200,
            "min_new_tokens": 1,
            "temperature": 0.5,
        }
        return WatsonxLLM(
            model_id="mistralai/mixtral-8x7b-instruct-v01",
            url=url,
            project_id=project_id,
            apikey=api_key,
            params=parameters
        )
    else:
        if api_key and project_id and url:
            print("Using custom WatsonX REST API for LLM...")
            return WatsonXRESTLLM(
                model_id="meta-llama/llama-3-3-70b-instruct",
                url=url,
                project_id=project_id,
                apikey=api_key
            )
        else:
            print("WARNING: WatsonX credentials not found. Using a dummy LLM.")
            from langchain_community.llms.fake import FakeListLLM
            return FakeListLLM(responses=[
                "I am a dummy AI because no API keys were provided in the .env file."
            ])


# Global variable to store the QA chain
qa_chain = None

def process_document(pdf_file):
    """
    Process the uploaded PDF document and create the QA bot pipeline.
    """
    global qa_chain
    if pdf_file is None:
        return "Please upload a PDF file first."

    # 1. Loading documents using LangChain
    loader = PyPDFLoader(pdf_file.name)
    documents = loader.load()

    # 2. Applying text splitting techniques
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=100
    )
    docs = text_splitter.split_documents(documents)

    # 3. Generating embeddings using watsonx embeddings
    embeddings = watsonx_embedding()

    # 4. Creating and configuring a vector database
    vectorstore = Chroma.from_documents(docs, embeddings)

    # 5. Developing a retriever for document querying
    retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

    # 6. Constructing a QA bot using LangChain and an LLM
    llm = get_llm()
    
    prompt_template = """You are a helpful AI assistant analyzing a document. 
Use the following pieces of context to answer the question at the end. 
If you cannot find the exact answer in the context, provide the best summary or most relevant information you can find from the context. Do not just say "I don't know".

Context:
{context}

Question: {question}
Answer:"""
    PROMPT = PromptTemplate(
        template=prompt_template, input_variables=["context", "question"]
    )
    
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        chain_type_kwargs={"prompt": PROMPT}
    )

    return f"Document '{os.path.basename(pdf_file.name)}' processed successfully! You can now ask questions."

def answer_question(question):
    """
    Answer the user's question using the RAG pipeline.
    """
    if not qa_chain:
        return "Please upload and process a document first."
    
    try:
        response = qa_chain.invoke({"query": question})
        return response['result']
    except Exception as e:
        import traceback
        return f"ERROR: {str(e)}\n\nDetails:\n{traceback.format_exc()}"

# --- UI Setup with Gradio ---
custom_theme = gr.themes.Monochrome(
    font=[gr.themes.GoogleFont("Inter"), "ui-sans-serif", "system-ui", "sans-serif"]
).set(
    body_background_fill="#000000",
    block_background_fill="#111111",
    block_border_width="1px",
    block_border_color="#333333",
    block_radius="16px",
    input_radius="12px",
    button_primary_background_fill="#ffffff",
    button_primary_background_fill_hover="#e5e5e5",
    button_primary_text_color="#000000",
)

css = """
body, html { height: 100vh; margin: 0; padding: 0; overflow: hidden; background-color: #000000 !important; color: #ffffff !important; }
gradio-app { height: 100vh !important; display: flex; flex-direction: column; }
.gradio-container { height: 100vh !important; overflow: hidden !important; padding-top: 1rem !important; }
#header { margin-bottom: 1rem; }
.textbox textarea { color: #ffffff !important; border-radius: 12px !important; }
label span { color: #ffffff !important; }
.file-preview, .upload-container { border-radius: 12px !important; }
"""

with gr.Blocks(theme=custom_theme, title="Document QA Bot", css=css) as demo:
    with gr.Column(elem_id="header"):
        gr.Markdown(
            """
            <div style="text-align: center;">
                <h1 style="font-size: 2rem; font-weight: 300; margin-bottom: 0; color: #ffffff;">Document QA Bot</h1>
                <p style="color: #aaaaaa; font-size: 1rem; margin-top: 0.5rem;">A smart, minimalist assistant for analyzing your PDFs.</p>
            </div>
            """
        )
    
    with gr.Row(equal_height=True):
        with gr.Column(scale=1, min_width=300):
            gr.Markdown("<h3 style='color: #ffffff; margin-bottom: 5px; font-weight: 400;'>1. Upload Document</h3>")
            pdf_input = gr.File(label="Select PDF File", file_types=[".pdf"])
            process_btn = gr.Button("Analyze Document", variant="primary")
            status_output = gr.Textbox(label="Status", interactive=False, show_label=True, placeholder="Waiting for document...", lines=1)
        
        with gr.Column(scale=2, min_width=400):
            gr.Markdown("<h3 style='color: #ffffff; margin-bottom: 5px; font-weight: 400;'>2. Ask Questions</h3>")
            question_input = gr.Textbox(
                label="Your Question", 
                placeholder="e.g., What is the main conclusion of this paper?",
                show_label=False,
                lines=1
            )
            ask_btn = gr.Button("Get Answer", variant="primary")
            answer_output = gr.Textbox(
                label="AI Response", 
                lines=9, 
                interactive=False,
                show_label=True,
                placeholder="The AI's answer will appear here..."
            )

    # Wire up the buttons
    process_btn.click(fn=process_document, inputs=[pdf_input], outputs=[status_output])
    ask_btn.click(fn=answer_question, inputs=[question_input], outputs=[answer_output])

if __name__ == "__main__":
    demo.launch(share=True)
