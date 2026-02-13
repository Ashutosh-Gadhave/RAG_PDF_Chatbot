# ---------- Imports ----------
import os
from dotenv import load_dotenv

from langchain_groq import ChatGroq
from langchain.memory import ConversationBufferMemory
from langchain.prompts import ChatPromptTemplate

from rag import create_retriever


# ---------- Load API Key ----------
load_dotenv()

# ---------- Initialize LLM ----------
llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0,
    streaming=False
)

# ---------- Initialize Retriever ----------
retriever = create_retriever("data/sample.pdf")

# ---------- Memory ----------
memory = ConversationBufferMemory(
    memory_key="chat_history",
    return_messages=True
)

# ---------- RAG Prompt ----------
prompt = ChatPromptTemplate.from_template("""
You are a helpful AI assistant.
Answer the question ONLY using the provided context.
If the answer is not in the context, say "I don't know."

Context:
{context}

Question:
{question}

Answer:
""")

# ---------- Main Loop ----------
if __name__ == "__main__":
    print("\nRAG Assistant Ready! (type 'exit' to quit)\n")

    while True:
        query = input("You: ")

        if query.lower() == "exit":
            break

        # Retrieve documents
        docs = retriever.invoke(query)
        context = "\n\n".join([doc.page_content for doc in docs])

        # Build prompt
        final_prompt = prompt.format(
            context=context,
            question=query
        )

        # Call LLM
        response = llm.invoke(final_prompt)

        print("\nAssistant:", response.content, "\n")
