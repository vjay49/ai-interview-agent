from langchain_community.vectorstores import FAISS
from langchain_openai.embeddings import OpenAIEmbeddings

def create_vector_store(chunks: list, openai_api_key: str):
    """
    Takes a list of text chunks, creates embeddings, and stores them in FAISS.
    """
    embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
    vector_store = FAISS.from_texts(chunks, embeddings)
    return vector_store