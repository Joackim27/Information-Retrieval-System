# src/app.py
import nest_asyncio
nest_asyncio.apply()

import streamlit as st
from src.helper import (
    get_pdf_text,
    get_text_chunks,
    get_vector_store,
    get_conversational_chain
)

# Set page config first
st.set_page_config(page_title="Information Retrieval")

def user_input(user_question):
    # Check if the conversation chain is initialized
    if st.session_state.conversation is None:
        st.error("Please process PDF files before asking questions.")
        return

    # Invoke the conversation chain
    response = st.session_state.conversation.invoke({"question": user_question})
    answer = response.get("answer", "No answer found.")

    # Append user input and answer to chat history
    st.session_state.chatHistory.append({"role": "user", "content": user_question})
    st.session_state.chatHistory.append({"role": "assistant", "content": answer})

    # Display the entire chat history
    for message in st.session_state.chatHistory:
        if message["role"] == "user":
            st.write(f"User: {message['content']}")
        else:
            st.write(f"Reply: {message['content']}")

def main():
    st.header("Information-Retrieval-System")

    user_question = st.text_input("Ask a Question from the PDF Files")

    # Initialize session state variables
    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chatHistory" not in st.session_state:
        st.session_state.chatHistory = []  # Initialize as empty list

    if user_question:
        user_input(user_question)
    
    with st.sidebar:
        st.title("Menu:")
        pdf_docs = st.file_uploader(
            "Upload your PDF Files and Click on the Submit & Process Button",
            accept_multiple_files=True
        )

        if st.button("Submit & Process"):
            with st.spinner("Processing..."):
                # Process PDFs
                raw_text = get_pdf_text(pdf_docs)
                text_chunks = get_text_chunks(raw_text)
                vector_store = get_vector_store(text_chunks)
                
                # Reset conversation and chat history
                st.session_state.conversation = get_conversational_chain(vector_store)
                st.session_state.chatHistory = []
                st.success("Processing complete!")

if __name__ == "__main__":
    main()