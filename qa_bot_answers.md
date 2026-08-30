# Assignment Answers

Here are the answers to the 11 questions from your assignment based on the IBM Watsonx and LangChain lab context:

**Question 1: Which LangChain class is used to load PDF documents in this lab?**
**Answer:** `PyPDFLoader`

**Question 2: Why is the embedding model obtained from watsonx_embedding() required when creating the vector database using Chroma?**
**Answer:** The embedding model is required to convert the text chunks from the documents into numerical vector representations (embeddings). These vectors allow the database to perform similarity searches to find the most relevant document chunks based on a user's query.

**Question 3: Why is a text splitter required after loading a PDF document using the .load() method in LangChain?**
**Answer:** A text splitter is required because large documents often exceed the maximum token limit (context window) of the LLM. Splitting the document into smaller, manageable chunks ensures that the most relevant pieces of information can be retrieved and passed to the LLM without exceeding its limits.

**Question 4: Which step ensures that document chunks can be retrieved using similarity search in the retriever pipeline?**
**Answer:** Converting the vector database into a retriever using the `.as_retriever()` method ensures that the document chunks can be retrieved using similarity search. 

**Question 5: Which of the following statements are true about the watsonx_embedding() function and the embedding configuration used in the lab for defining embedding model?**
**Answer:** (Since the options are not provided, here is the general truth): The `watsonx_embedding()` function requires IBM Cloud credentials (such as an API key, URL, and Project ID) and specifies a particular embedding model ID (like `ibm/slate-125m-english-rtrvr`) to generate the vector embeddings.

**Question 6: Which method is used to convert a vector database into a retriever?**
**Answer:** `as_retriever()` (or `.as_retriever()`)

**Question 7: Which LangChain component is responsible for performing question answering using retrieval-augmented generation (RAG)?**
**Answer:** `RetrievalQA` (or the newer `create_retrieval_chain` / `create_stuff_documents_chain` combination, but most IBM labs use `RetrievalQA.from_chain_type`).

**Question 8: Which of the following options represent the correct logical sequence of steps involved in building the QA bot in the lab?**
**Answer:** 
1. Load Document
2. Text Splitting
3. Generate Embeddings
4. Create Vector DB
5. Create Retriever
6. Construct QA Bot

**Question 9: What is the role of the get_llm() function in the QA bot pipeline?**
**Answer:** The role of the `get_llm()` function is to instantiate and configure the IBM Watsonx Large Language Model (LLM) with specific generation parameters (like decoding method, max new tokens, temperature, etc.) so it can generate answers based on the retrieved context.

**Question 10: Upload a screenshot named QA_bot.png that clearly shows the QA bot interface you created.**
**Answer:** See the instructions below on how to run the provided Python application to generate this interface and take your screenshot.

**Question 11: Which of the following components are used in the QA bot implementation to interact with the user?**
**Answer:** **Gradio** (specifically `gradio` components like `gr.Interface` or `gr.Blocks`, which are standard in these labs for building the web interface).
