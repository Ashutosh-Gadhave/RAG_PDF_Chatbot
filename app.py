import streamlit as st
from rag import create_retriever
from langchain_groq import ChatGroq
from dotenv import load_dotenv
import os

# ---------- Load Environment ----------
load_dotenv()

# ---------- Page Config ----------
st.set_page_config(
    page_title="RAG Assistant",
    page_icon="📄",
    layout="wide"
)

# ---------- Sidebar ----------
st.sidebar.title("⚙️ Settings")
st.sidebar.markdown("Upload a PDF and start chatting.")

if st.sidebar.button("🗑️ Clear Chat"):
    st.session_state.messages = []

# ---------- Title ----------
st.title("📄 RAG Document Assistant")
st.write("Ask questions grounded in your uploaded document.")

# ---------- Upload ----------
uploaded_file = st.file_uploader("Upload PDF", type="pdf")

if uploaded_file:

    with st.spinner("Processing document..."):

        with open("temp.pdf", "wb") as f:
            f.write(uploaded_file.read())

        retriever = create_retriever("temp.pdf")

        llm = ChatGroq(
            model="llama-3.3-70b-versatile",
            temperature=0
        )

    # Initialize chat memory
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat history
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # User input
    if prompt := st.chat_input("Ask a question about the document..."):

        st.session_state.messages.append({"role": "user", "content": prompt})

        with st.chat_message("user"):
            st.markdown(prompt)

        with st.spinner("Thinking..."):

            docs = retriever.invoke(prompt)

            if docs:
                context = "\n\n".join([doc.page_content for doc in docs])
            else:
                context = "No relevant information found."

            response = llm.invoke(
                f"""
                You are a helpful AI assistant.
                Answer ONLY using the context below.
                If answer is not found, say "Not found in document."

                Context:
                {context}

                Question:
                {prompt}
                """
            )

            answer = response.content

        st.session_state.messages.append(
            {"role": "assistant", "content": answer}
        )

        with st.chat_message("assistant"):
            st.markdown(answer)

        # 🔥 Show sources (interview gold)
        with st.expander("📚 Source Chunks"):
            for i, doc in enumerate(docs):
                st.write(f"Chunk {i+1}:")
                st.write(doc.page_content[:500])
