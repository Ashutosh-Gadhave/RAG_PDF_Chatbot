# rag.py

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings


def create_retriever(pdf_path):

    loader = PyPDFLoader(pdf_path)
    documents = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

    docs = text_splitter.split_documents(documents)

    embedding = HuggingFaceEmbeddings()

    vectorstore = FAISS.from_documents(docs, embedding)

    # 🔥 IMPORTANT CHANGE
    return vectorstore.as_retriever(search_kwargs={"k": 3})
