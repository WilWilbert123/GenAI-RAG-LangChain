import os
os.environ["HF_HOME"] = os.path.abspath("./hf_cache")
import gradio as gr
from fastapi import FastAPI
from dotenv import load_dotenv

# Langchain imports
import requests
from typing import Any, List, Mapping, Optional
from langchain.llms.base import LLM
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, TextLoader, CSVLoader
from langchain.docstore.document import Document
import pandas as pd
import pytesseract
from PIL import Image
import whisper
import moviepy.editor as mp
import warnings
warnings.filterwarnings("ignore", message=".*FP16 is not supported on CPU.*")
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
                "max_new_tokens": 1500,
                "min_new_tokens": 1,
                "temperature": 0.5,
                "stop_sequences": stop if stop else ["\nQuestion:", "Now, I have another question", "\nNow,"]
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
            "max_new_tokens": 1500,
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
        return "Please upload a document first.", "<div style='text-align:center; color:#555;'>No file uploaded</div>"

    preview_html = "<div style='text-align:center; color:#555;'>Preview not available</div>"

    # 1. Loading documents using LangChain
    ext = os.path.splitext(pdf_file.name)[1].lower()
    if ext == '.pdf':
        preview_html = f'<iframe src="/file={pdf_file.name}" width="100%" height="200px" style="border:none; border-radius:8px;"></iframe>'
        loader = PyPDFLoader(pdf_file.name)
        documents = loader.load()
    elif ext == '.docx':
        preview_html = "<div style='padding:10px; color:#aaa;'>[Word Document Uploaded - Preview Not Supported]</div>"
        loader = Docx2txtLoader(pdf_file.name)
        documents = loader.load()
    elif ext == '.txt':
        try:
            with open(pdf_file.name, 'r', encoding='utf-8') as f:
                snippet = f.read(500)
            preview_html = f'<div style="max-height:200px; overflow:auto; white-space:pre-wrap; font-size:12px; padding:10px; background:#111; border-radius:8px;">{snippet}...</div>'
        except:
            pass
        loader = TextLoader(pdf_file.name)
        documents = loader.load()
    elif ext == '.csv':
        try:
            df = pd.read_csv(pdf_file.name)
            preview_html = f'<div style="max-height:200px; overflow:auto; background:#111; color:#fff; padding:5px; font-size:12px; border-radius:8px;">{df.head(5).to_html()}</div>'
        except:
            pass
        loader = CSVLoader(pdf_file.name)
        documents = loader.load()
    elif ext in ['.xlsx', '.xls']:
        try:
            df = pd.read_excel(pdf_file.name)
            text_content = df.to_string()
            preview_html = f'<div style="max-height:200px; overflow:auto; background:#111; color:#fff; padding:5px; font-size:12px; border-radius:8px;">{df.head(5).to_html()}</div>'
        except:
            text_content = ""
        documents = [Document(page_content=text_content, metadata={"source": pdf_file.name})]
    elif ext in ['.png', '.jpg', '.jpeg', '.gif']:
        preview_html = f'<div style="display:flex; justify-content:center;"><img src="/file={pdf_file.name}" style="max-width:100%; max-height:200px; object-fit:contain; border-radius:8px;" /></div>'
        # OCR for images
        try:
            text_content = pytesseract.image_to_string(Image.open(pdf_file.name))
        except Exception as e:
            return f"OCR failed: {str(e)}. (Ensure tesseract is installed via Homebrew)", preview_html
        
        import re
        if len(re.findall(r'[a-zA-Z0-9]', text_content)) < 5:
            return "No meaningful text could be found in this image. The bot can only read text documents, not pure images/logos.", preview_html
        documents = [Document(page_content=text_content, metadata={"source": pdf_file.name})]
    elif ext in ['.mp3', '.mp4']:
        if ext == '.mp4':
            preview_html = f'<div style="display:flex; justify-content:center;"><video controls src="/file={pdf_file.name}" style="max-width:100%; max-height:200px; border-radius:8px;"></video></div>'
        else:
            preview_html = f'<div style="display:flex; justify-content:center; padding: 20px;"><audio controls src="/file={pdf_file.name}"></audio></div>'
            
        audio_path = pdf_file.name
        if ext == '.mp4':
            # Extract audio
            audio_path = pdf_file.name + ".mp3"
            try:
                clip = mp.VideoFileClip(pdf_file.name)
                clip.audio.write_audiofile(audio_path, logger=None)
            except Exception as e:
                return f"Failed to extract audio from video: {str(e)}", preview_html
        
        # Transcribe with Whisper
        try:
            model = whisper.load_model("base")
            result = model.transcribe(audio_path)
            text_content = result["text"]
        except Exception as e:
            return f"Audio transcription failed: {str(e)}. (Ensure ffmpeg is installed via Homebrew)", preview_html
            
        if not text_content.strip():
            return "No speech could be found in this media file.", preview_html
        documents = [Document(page_content=text_content, metadata={"source": pdf_file.name})]
    else:
        return f"Unsupported file type: {ext}", preview_html

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
If the context is empty, contains only junk characters, or does not contain enough information, simply reply with: "The uploaded document does not contain enough information to answer this question." Do NOT generate a long list of reasons and do NOT repeat yourself.
Otherwise, provide only the answer and nothing else. Do not ask follow-up questions.
Format your answer clearly: use bullet points and separate paragraphs to make it highly organized and easy to read.

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

    return f"Document '{os.path.basename(pdf_file.name)}' processed successfully! You can now ask questions.", preview_html

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
    body_background_fill="#f5f5f5",
    body_background_fill_dark="#000000",
    block_background_fill="#ffffff",
    block_background_fill_dark="#111111",
    block_border_color="#dddddd",
    block_border_color_dark="#333333",
    block_border_width="1px",
    block_radius="16px",
    input_radius="12px",
    body_text_color="#000000",
    body_text_color_dark="#ffffff",
    button_primary_background_fill="#000000",
    button_primary_background_fill_dark="#ffffff",
    button_primary_text_color="#ffffff",
    button_primary_text_color_dark="#000000",
)

css = """
/* Base styles */
body, html { margin: 0; padding: 0; }
gradio-app { display: flex; flex-direction: column; }
.gradio-container { max-width: 100% !important; padding: 0.2rem 1rem !important; position: relative; }
#header { margin-bottom: 0.2rem; }
h3 { margin-top: 2px !important; margin-bottom: 2px !important; font-size: 1rem !important; }
.upload-container, .file-preview { min-height: 80px !important; max-height: 80px !important; display: flex !important; justify-content: center !important; align-items: center !important; }
.upload-container > div { padding: 0 !important; }
footer { display: none !important; }

/* Theme Toggle Button */
#theme-toggle {
    position: absolute;
    top: 15px;
    left: 15px;
    cursor: pointer;
    z-index: 1000;
    background: none;
    border: none;
    padding: 0;
    color: var(--body-text-color);
}

/* Fix for Examples (Tables) text color */
.dark table, .dark th, .dark td { color: #ffffff !important; }
body:not(.dark) table, body:not(.dark) th, body:not(.dark) td { color: #000000 !important; }

"""

with gr.Blocks(theme=custom_theme, title="Document QA Bot", css=css) as demo:
    gr.HTML("""
        <button id="theme-toggle" onclick="
            document.body.classList.toggle('dark');
            const isDark = document.body.classList.contains('dark');
            const sun = '<svg xmlns=\\'http://www.w3.org/2000/svg\\' width=\\'20\\' height=\\'20\\' viewBox=\\'0 0 24 24\\' fill=\\'none\\' stroke=\\'currentColor\\' stroke-width=\\'2\\' stroke-linecap=\\'round\\' stroke-linejoin=\\'round\\'><circle cx=\\'12\\' cy=\\'12\\' r=\\'5\\'></circle><line x1=\\'12\\' y1=\\'1\\' x2=\\'12\\' y2=\\'3\\'></line><line x1=\\'12\\' y1=\\'21\\' x2=\\'12\\' y2=\\'23\\'></line><line x1=\\'4.22\\' y1=\\'4.22\\' x2=\\'5.64\\' y2=\\'5.64\\'></line><line x1=\\'18.36\\' y1=\\'18.36\\' x2=\\'19.78\\' y2=\\'19.78\\'></line><line x1=\\'1\\' y1=\\'12\\' x2=\\'3\\' y2=\\'12\\'></line><line x1=\\'21\\' y1=\\'12\\' x2=\\'23\\' y2=\\'12\\'></line><line x1=\\'4.22\\' y1=\\'19.78\\' x2=\\'5.64\\' y2=\\'18.36\\'></line><line x1=\\'18.36\\' y1=\\'5.64\\' x2=\\'19.78\\' y2=\\'4.22\\'></line></svg>';
            const moon = '<svg xmlns=\\'http://www.w3.org/2000/svg\\' width=\\'20\\' height=\\'20\\' viewBox=\\'0 0 24 24\\' fill=\\'none\\' stroke=\\'currentColor\\' stroke-width=\\'2\\' stroke-linecap=\\'round\\' stroke-linejoin=\\'round\\'><path d=\\'M21 12.79A9 9 0 1 1 11.21 3 7 7 0 0 0 21 12.79z\\'></path></svg>';
            this.innerHTML = isDark ? sun : moon;
        ">
        <svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="5"></circle><line x1="12" y1="1" x2="12" y2="3"></line><line x1="12" y1="21" x2="12" y2="23"></line><line x1="4.22" y1="4.22" x2="5.64" y2="5.64"></line><line x1="18.36" y1="18.36" x2="19.78" y2="19.78"></line><line x1="1" y1="12" x2="3" y2="12"></line><line x1="21" y1="12" x2="23" y2="12"></line><line x1="4.22" y1="19.78" x2="5.64" y2="18.36"></line><line x1="18.36" y1="5.64" x2="19.78" y2="4.22"></line></svg>
        </button>
    """)
    with gr.Column(elem_id="header"):
        gr.Markdown(
            """
            <div style="text-align: center;">
                <h1 style="font-size: 1.3rem; font-weight: 300; margin-bottom: 0; color: var(--body-text-color);">Document QA Bot</h1>
                <p style="font-size: 0.8rem; margin-top: 0.1rem; color: var(--body-text-color); opacity: 0.7;">A smart, minimalist assistant for analyzing your documents, images, and media.</p>
            </div>
            """
        )
    
    with gr.Row():
        with gr.Column(scale=1, min_width=300):
            gr.Markdown("<h3 style='font-weight: 400; color: var(--body-text-color);'>1. Upload Document (Auto-Analyzes)</h3>")
            pdf_input = gr.File(label="Select Document or Media File", file_types=[".pdf", ".docx", ".txt", ".csv", ".xlsx", ".xls", ".png", ".jpg", ".jpeg", ".gif", ".mp4", ".mp3"])
            status_output = gr.Textbox(label="Status", interactive=False, show_label=True, placeholder="Waiting for document...", lines=1)
            preview_area = gr.HTML(value="<div style='text-align:center; color:#555; padding: 10px; border: 1px dashed #333; border-radius: 8px;'>No preview available</div>")
            
            gr.Markdown("<h3 style='font-weight: 400; color: var(--body-text-color);'>2. Ask Questions</h3>")
            question_input = gr.Textbox(
                label="Your Question", 
                placeholder="e.g., What is the main conclusion of this paper?",
                show_label=False,
                lines=1
            )
            gr.Examples(
                examples=[
                    "What is the main conclusion of this document?",
                    "Can you summarize the key points?",
                    "What are the main topics discussed?",
                    "What are the key takeaways from this text?",
                    "Can you explain the main argument presented?",
                    "What specific details or data are mentioned?"
                ],
                inputs=question_input
            )
            
        with gr.Column(scale=1, min_width=300):
            gr.Markdown("<h3 style='font-weight: 400; color: var(--body-text-color);'>AI Response</h3>")
            answer_output = gr.Textbox(
                label="AI Response", 
                lines=15, 
                interactive=False,
                show_label=False,
                placeholder="The AI's answer will appear here in real-time..."
            )

    # Wire up the events
    pdf_input.change(fn=process_document, inputs=[pdf_input], outputs=[status_output, preview_area])
    question_input.change(fn=answer_question, inputs=[question_input], outputs=[answer_output])
    question_input.submit(fn=answer_question, inputs=[question_input], outputs=[answer_output])
    
    # Footer
    gr.HTML(
        """
        <div style='text-align: center; font-size: 0.8rem; margin-top: 1rem; margin-bottom: 0.5rem; display: flex; justify-content: center; align-items: center; gap: 10px; color: var(--body-text-color); opacity: 0.7;'>
            <span>Wilbert Gamis 2026</span>
            <span>&bull;</span>
            <a href='https://github.com/WilWilbert123/GenAI-RAG-LangChain' target='_blank' style='text-decoration: none; color: inherit; display: flex; align-items: center; gap: 4px;'>
                <svg height="16" width="16" viewBox="0 0 16 16" fill="currentColor">
                    <path d="M8 0C3.58 0 0 3.58 0 8c0 3.54 2.29 6.53 5.47 7.59.4.07.55-.17.55-.38 0-.19-.01-.82-.01-1.49-2.01.37-2.53-.49-2.69-.94-.09-.23-.48-.94-.82-1.13-.28-.15-.68-.52-.01-.53.63-.01 1.08.58 1.23.82.72 1.21 1.87.87 2.33.66.07-.52.28-.87.51-1.07-1.78-.2-3.64-.89-3.64-3.95 0-.87.31-1.59.82-2.15-.08-.2-.36-1.02.08-2.12 0 0 .67-.21 2.2.82.64-.18 1.32-.27 2-.27.68 0 1.36.09 2 .27 1.53-1.04 2.2-.82 2.2-.82.44 1.1.16 1.92.08 2.12.51.56.82 1.27.82 2.15 0 3.07-1.87 3.75-3.65 3.95.29.25.54.73.54 1.48 0 1.07-.01 1.93-.01 2.2 0 .21.15.46.55.38A8.013 8.013 0 0016 8c0-4.42-3.58-8-8-8z"></path>
                </svg>
                GitHub
            </a>
            <span>&bull;</span>
            <span>Use via API</span>
            <span>&bull;</span>
            <span>Built with Gradio</span>
        </div>
        """
    )

# Vercel requires an ASGI app to be exposed as "app" in app.py
app = FastAPI()
app = gr.mount_gradio_app(app, demo, path="/")

if __name__ == "__main__":
    demo.launch(share=True)
