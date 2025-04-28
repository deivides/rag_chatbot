import os
import tempfile
import streamlit as st

from decouple import config

from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings


st.set_page_config(
    page_title='Chatbot',
    page_icon='🤖',
)

os.environ["OPENAI_API_KEY"] = config('OPENAI_API_KEY')
persist_directory = 'db'


def process_pdf(file):
    with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as temp_file:
        temp_file.write(file.read())
        temp_file_path = temp_file.name

    loader = PyPDFLoader(temp_file_path)
    docs = loader.load()
    os.remove(temp_file_path)

    text_spliter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150,
    )
    chunks = text_spliter.split_documents(documents=docs)
    return chunks


@st.cache_resource
def load_existing_vector_store():
    print("Loading or reloading the vector store...")
    if os.path.exists(os.path.join(persist_directory)):
        vector_store = Chroma(
            persist_directory=persist_directory,
            embedding_function=OpenAIEmbeddings(),
        )
        print(f"Vector store loaded with {vector_store._collection.count()} documents.")
        return vector_store
    else:
        print("No existing vector store found.")
        return None


def add_to_vector_store(chunks, vector_store=None):
    print("Adding documents to the vector store...")
    if vector_store:
        vector_store.add_documents(chunks)
        print(f"{len(chunks)} documents added to the existing vector store.")
    else:
        vector_store = Chroma.from_documents(
            documents=chunks,
            embedding=OpenAIEmbeddings(),
            persist_directory=persist_directory,
        )
        print(f"New vector store created with {len(chunks)} documents.")
    return vector_store


def ask_question(model, query, vector_store):
    print(f"Asking the model: {query}")
    llm = ChatOpenAI(model=model)
    retriever = vector_store.as_retriever()

    system_prompt = '''
    Use the context to answer the questions.
    If you can't find an answer in the context,
    explain that there is no information available.
    Answer in markdown format with detailed and
    interactive visualizations.
    Context: {context}
    '''
    messages = [('system', system_prompt)]
    for message in st.session_state.messages:
        messages.append((message.get('role'), message.get('content')))
    messages.append(('human', '{input}'))

    prompt = ChatPromptTemplate.from_messages(messages)

    question_answer_chain = create_stuff_documents_chain(
        llm=llm,
        prompt=prompt,
    )
    chain = create_retrieval_chain(
        retriever=retriever,
        combine_docs_chain=question_answer_chain,
    )
    response = chain.invoke({'input': query})
    print(f"Model's response: {response.get('answer')}")
    return response.get('answer')


vector_store = load_existing_vector_store()


st.header('🤖 Chat with your documents using RAG')


with st.sidebar:
    st.header('Upload files')
    uploaded_files = st.file_uploader(
        label='Upload PDF files',
        type=['pdf'],
        accept_multiple_files=True,
    )
    st.info('🔒 Your files are processed locally. No data is sent to external server.')

    if uploaded_files:
        if st.button('Process files'):
            with st.spinner('Reading your files...'):
                all_chunks = []
                for uploaded_file in uploaded_files:
                    chunks = process_pdf(file=uploaded_file)
                    all_chunks.extend(chunks)
                vector_store = add_to_vector_store(
                    chunks=all_chunks,
                    vector_store=vector_store,
                )
                st.success('Files processed successfully!')
                st.cache_resource.clear()
                vector_store = load_existing_vector_store()
                if vector_store:
                    st.write(f"Number of documents in vector store: {vector_store._collection.count()}")

    model_options = [
        'gpt-3.5-turbo',
        'gpt-4',
        'gpt-4-turbo',
        'gpt-4o-mini',
        'gpt-4o',
    ]
    selected_model = st.sidebar.selectbox(
        label='Select a large language model',
        options=model_options,
    )

if 'messages' not in st.session_state:
    st.session_state['messages'] = []

question = st.chat_input('How can I help you today?')

if vector_store and question:
    for message in st.session_state.messages:
        st.chat_message(message.get('role')).write(message.get('content'))

    st.chat_message('user').write(question)
    st.session_state.messages.append({'role': 'user', 'content': question})

    with st.spinner('Finding the best answer...'):
        response = ask_question(
            model=selected_model,
            query=question,
            vector_store=vector_store,
        )

        st.chat_message('ai').write(response)
        st.session_state.messages.append({'role': 'ai', 'content': response})
