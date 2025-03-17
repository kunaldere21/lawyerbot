import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_ollama import OllamaEmbeddings

load_dotenv()
def get_llm():
    llm_model = ChatGroq(model='deepseek-r1-distill-llama-70b')
    return llm_model

# Load database
def get_vectorstore():
    db_path = 'vectordb/db_faiss'
    embedding_model =OllamaEmbeddings(model='deepseek-r1:7b')
    db = FAISS.load_local(db_path, embedding_model, allow_dangerous_deserialization=True)
    return db

# retrieve doc
def retrieve_docs(query):
    faiss_db = get_vectorstore()
    return faiss_db.similarity_search(query)

def get_context(documnets):
    context = "\n\n".join([doc.page_content for doc in documnets])
    return context


# Answer Question
def answer_query(documents, query):
    model = get_llm()
    custom_prompt_template = """
    Use the pieces of information provided in the context to answer user's question.
    Answer should be structure formate.
    If you dont know the answer, just say that you dont know, dont try to make up an answer. 
    Dont provide anything out of the given context
    Question: {question} 
    Context: {context} 
    Answer:
    """
    prompt = PromptTemplate(
        input_variables=['question','context'],
        template=custom_prompt_template
    )
    context = get_context(documents)
    chain = prompt | model
    result = chain.invoke({'question': query, 'context':context})
    return result



if __name__== "__main__":
    
    USER_QUERY = input("Ask something: ")


    documents = retrieve_docs(USER_QUERY)

    response = answer_query(documents, USER_QUERY)
    print(response.content)



    




