import streamlit as st
from rag_pipeline import answer_query, retrieve_docs, get_vectorstore

upload_file = st.file_uploader("upload pdf",
                               type= 'pdf',
                               accept_multiple_files=False)

# step 2- chatbot skeleton (QnA)

user_query = st.text_area("Enter your prompt: ", height=150, placeholder="Ask Anything")


ask_question = st.button("Ask AI Lawyer")
if ask_question:
    if upload_file:
        st.chat_message("user").write(user_query)

    
        documents = retrieve_docs(user_query)

        response = answer_query(documents=documents, query=user_query)

        st.chat_message("AI Lawyer").write(response.content)

    else:
        st.error("Kindly, upload valid pdf file")