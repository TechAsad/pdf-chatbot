from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores.faiss import FAISS
from langchain_community.document_loaders import DirectoryLoader
from langchain.tools import tool


class VectorSearchTools():

  @tool("Search the vector database")
  def dbsearch(query):
        """
        useful to search vector database and returns most relevant chunks
        """
        # Processing PDF and DOCX files
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={'device': 'cpu'})
        db = FAISS.load_local("faiss", embeddings)
        
        retrieved_docs = db.similarity_search(query, k=5, fetch_k= 40)



        return retrieved_docs
