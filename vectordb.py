from langchain_community.document_loaders import DirectoryLoader, PDFPlumberLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaEmbeddings




def upload_pdf():
    pass

def load_pdf(file_path):
    loader = PDFPlumberLoader(file_path)
    documnents = loader.load()
    return documnents

# create chunks
def create_cunks(documents):
    text_splitter= RecursiveCharacterTextSplitter(
        chunk_size = 1000,
        chunk_overlap= 200,
        add_start_index= True
    )
    text_chunks = text_splitter.split_documents(documents)
    return text_chunks
    

# setup embedding model deepseek r1 with ollama
def get_embbeding(model_name):
    embedding_model = OllamaEmbeddings(model= model_name)
    return embedding_model

def vectordb_store(text_chunks, embedding_model, db_path):
    faiss_db = FAISS.from_documents(text_chunks, embedding_model)
    faiss_db.save_local(db_path)
    return faiss_db


if __name__ =="__main__":
    # upload and load raw pdf file
    pdfs_directory = "pdfs/"

    file_path = 'pdfs/udhr_booklet_en_web.pdf'
    documents = load_pdf(file_path)
    print("count of documnets: ", len(documents))

    text_chunks = create_cunks(documents)
    print("chunk count: ", len(text_chunks))


    ollama_model_name = 'deepseek-r1:7b'
    embedding_model = get_embbeding(ollama_model_name)

    # vectordb
    FAISS_DB_PATH = 'vectordb/db_faiss'
    faiss_db = vectordb_store(text_chunks, embedding_model, FAISS_DB_PATH)



