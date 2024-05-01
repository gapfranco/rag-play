import os
from pathlib import Path
from dotenv import load_dotenv

import streamlit as st
from langchain_anthropic import ChatAnthropic
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI

from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

from langchain.schema import (
    HumanMessage,
    AIMessage
)
from langchain.text_splitter import RecursiveCharacterTextSplitter as Splitter

from PyPDF2 import PdfReader

LOCAL_VECTOR_DIR = Path(__file__).resolve().parent.joinpath('data',
                                                            'vector_store')
load_dotenv()
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY")


def init_page():
    st.set_page_config(
        page_title="RAG App",
        page_icon="🦜"
    )
    st.header("🦜🔗 Langchain RAG playground")
    st.sidebar.title("Opções")
    st.session_state.source_docs = []
    st.session_state.retriever = embeddings_on_local_vectordb()


def init_messages():
    clear_button = st.sidebar.button("Limpar conversas",
                                     on_click=clear_messages)
    st.session_state.source_docs = st.file_uploader(
        "Upload documentos",
        accept_multiple_files=True,
        type="pdf"
    )
    st.button("Enviar Documentos", on_click=process_documents)
    if "messages" not in st.session_state:
        st.session_state.messages = []


def clear_messages():
    st.session_state.messages = []


def process_documents():
    documents = ""
    try:
        for source_doc in st.session_state.source_docs:
            documents += load_pdf(source_doc)
        texts = split_documents(documents)
        st.session_state.vectordb.add_texts(texts)

    except Exception as e:
        st.error(f"Ocorreu umm erro: {e}")


def load_pdf(source_doc):
    pdf_reader = PdfReader(source_doc)
    pages = [page.extract_text() for page in pdf_reader.pages]
    text = "\n\n".join([page for page in pages if page])
    return text


def split_documents(documents):
    txt_split = Splitter.from_tiktoken_encoder(
        model_name="text-embedding-ada-002",
        # The appropriate chunk size needs to be adjusted based
        # on the PDF being queried.
        # If it's too large, it may not be able to reference
        # information from
        # various parts during question answering.
        # On the other hand, if it's too small, one chunk may
        # not contain enough contextual information.
        chunk_size=500,
        chunk_overlap=0,
    )
    texts = txt_split.split_text(documents)
    return texts


def embeddings_on_local_vectordb():
    vectordb = Chroma(embedding_function=OpenAIEmbeddings(),
                      persist_directory=LOCAL_VECTOR_DIR.as_posix())
    st.session_state.vectordb = vectordb
    retriever = vectordb.as_retriever(
        # There are also "mmr," "similarity_score_threshold," and others.
        search_type="similarity",
        # How many documents to retrieve? (default: 4)
        search_kwargs={"k": 10},
    )
    return retriever


def select_model():
    model_options = []
    if OPENAI_API_KEY:
        model_options.append("OpenAI gpt-3.5-turbo")
        model_options.append("OpenAI gpt-4")
    if GOOGLE_API_KEY:
        model_options.append("Google gemini-pro")
    if ANTHROPIC_API_KEY:
        model_options.append("Anthropic claude-3-opus-20240229")
        model_options.append("Anthropic claude-3-haiku-20240307")
    model_opt = st.sidebar.selectbox("Escolha um modelo:", model_options)
    model = None
    llm = None
    if model_opt:
        model = model_opt.split()[1]
    if model in {"gpt-3.5-turbo", "gpt-4"}:
        llm = ChatOpenAI(temperature=0, model_name=model)
    elif model in {"gemini-pro"}:
        llm = ChatGoogleGenerativeAI(model=model)
    elif model in {"claude-3-opus-20240229", "claude-3-haiku-20240307"}:
        llm = ChatAnthropic(temperature=0, model_name=model)
    return llm


def query_llm(retriever, query, llm):
    qa_chain = get_retrieval_lcel(retriever, llm)
    result = qa_chain.invoke(query)
    st.session_state.messages.append((query, result))
    return result


def get_retrieval_lcel(retriever, llm):
    template = """Use o que sabe e mais os seguintes trechos de contexto para 
    responder à pergunta no final. 
    Use até dez sentenças no máximo e mantenha a resposta o mais detalhada
    possível.

    {context}

    Pergunta: {question}

    Resposta:"""

    prompt = PromptTemplate.from_template(template)

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    rag_chain = (
            {
                "context": retriever | format_docs,
                "question": RunnablePassthrough(),
            }
            | prompt
            | llm
            | StrOutputParser()
    )
    return rag_chain


def main():
    init_page()
    model = select_model()
    init_messages()

    # Monitor user input
    if user_input := st.chat_input("Pergunte aqui:"):
        st.session_state.messages.append(HumanMessage(content=user_input))
        if st.session_state.get("retriever"):
            with st.spinner("Pesquisando..."):
                response = query_llm(st.session_state.retriever, user_input,
                                     model
                                     )
            st.session_state.messages.append(AIMessage(content=response))

    messages = st.session_state.get('messages', [])
    for message in messages:
        if isinstance(message, AIMessage):
            with st.chat_message('assistant'):
                st.markdown(message.content)
        elif isinstance(message, HumanMessage):
            with st.chat_message('user'):
                st.markdown(message.content)
        # else:  # isinstance(message, SystemMessage):
        #     st.write(f"System message: {message.content}")


if __name__ == '__main__':
    main()
