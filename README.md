# 🤖 Chatbot with RAG and Multiple LLM Support

This project implements a chatbot that leverages the power of **Retrieval-Augmented Generation (RAG)** to answer questions based on user-uploaded PDF documents. It utilizes multiple Large Language Models (LLMs) from OpenAI, facilitated by the **Langchain** framework, and provides a user-friendly interface built with **Streamlit**.

## ✨ Features

* **Document Upload:** Easily upload multiple PDF files to be used as the knowledge base.
* **RAG Implementation:** Employs Retrieval-Augmented Generation to provide contextually relevant answers by combining information retrieval with LLM generation.
* **Multiple LLM Support:** Choose between various OpenAI models (`gpt-3.5-turbo`, `gpt-4`, `gpt-4-turbo`, `gpt-4o-mini`, `gpt-4o`) to experiment with different response styles and capabilities.
* **Vector Database:** Creates and utilizes a vectorized **SQLite** database (using ChromaDB) to store embeddings of the document chunks for efficient semantic search. 💾
* **Contextual Conversations:** Maintains the history of your conversation, allowing for more natural and follow-up questions. 💬
* **User-Friendly Interface:** Built with Streamlit for an intuitive and interactive experience. streamlit
* **Local Processing:** Your files are processed locally, ensuring data privacy. 🔒

## 🛠️ Technologies Used

* **Python:** The primary programming language.
* **Langchain:** A framework for building LLM-powered applications. 🔗
* **Streamlit:** A library for creating interactive web applications. streamlit
* **OpenAI API:** For accessing Large Language Models and embedding models. 🔑
* **`decouple`:** To manage API keys and other configurations securely.
* **`PyPDFLoader` (from `langchain_community.document_loaders`):** To load and process PDF files.
* **`RecursiveCharacterTextSplitter` (from `langchain.text_splitter`):** To split documents into manageable chunks.
* **`Chroma` (from `langchain_chroma`):** For creating and managing the vector database. 💾
* **`OpenAIEmbeddings` (from `langchain_openai`):** To generate vector embeddings of the text chunks.
* **`ChatPromptTemplate` (from `langchain_core.prompts`):** To create structured prompts for the LLMs.
* **`create_stuff_documents_chain` (from `langchain.chains.combine_documents`):** To combine retrieved documents with the question for the LLM.
* **`create_retrieval_chain` (from `langchain.chains.retrieval`):** To create the end-to-end retrieval and generation chain.

## ⚙️ Setup

1.  **Clone the repository** (if applicable).
2.  **Install the required libraries:**
    ```bash
    pip install -r requirements.txt
    ```
3.  **Set up your OpenAI API key:**
    * Create a `.env` file in the project root.
    * Add your OpenAI API key to the `.env` file:
        ```
        OPENAI_API_KEY="YOUR_OPENAI_API_KEY"
        ```
4.  **Run the Streamlit application:**
    ```bash
    streamlit run app.py 
    ```

## 💡 How it Works

1.  **Document Processing:** When you upload PDF files, the application processes them by:
    * Loading the text content.
    * Splitting the text into smaller chunks with some overlap to maintain context. ✂️
    * Generating vector embeddings for each chunk using OpenAI's embedding model. 📐
    * Storing these embeddings in a Chroma vector database (backed by SQLite) on your local machine. 💾
2.  **Question Answering:** When you ask a question:
    * The question is also converted into a vector embedding. 📐
    * The vector database is queried to find the most relevant document chunks based on semantic similarity. 🚀
    * The retrieved chunks, along with your question and the chat history, are passed to the selected OpenAI LLM. 🧠
    * The LLM generates a response based on the provided context.
3.  **Chat History:** The application keeps track of the conversation history, allowing the LLM to consider previous turns when generating the next response, leading to more coherent and contextual interactions. 💬

## 📂 Directory Structure (Example)

Markdown

.
├── .env
├── db/           # Directory where the Chroma vector database is stored (SQLite files) 💾
├── requirements.txt
└── your_script_name.py  # Your main Streamlit application file


## 🚀 Usage

1.  Open the Streamlit application in your web browser.
2.  Upload one or more PDF files using the file uploader in the sidebar.
3.  Click the "Process files" button to index the documents.
4.  Select your preferred OpenAI LLM from the dropdown in the sidebar.
5.  Start asking questions in the chat input field!

## Disclaimer

* This application processes your files locally. No data is sent to external servers for processing, except for the interaction with the OpenAI API for embeddings and LLM responses.
* Ensure you have a valid OpenAI API key to use the LLM functionalities.

## Contributions

Contributions are welcome! Feel free to open issues or submit pull requests.
