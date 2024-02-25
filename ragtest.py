from langchain.vectorstores import Chroma
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
import sentence_transformers

from langchain.document_loaders import UnstructuredFileIOLoader, UnstructuredFileLoader  
from langchain.text_splitter import RecursiveCharacterTextSplitter

loader = UnstructuredFileLoader('./inputs/rag.txt')
data = loader.load()
print(data)
texp_splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=0)  # 100 characters per chunk with 0 overlap 中文： 100个字，不重叠
split_docs = texp_splitter.split_documents(data)
print(split_docs)

embedding_model_dict = {
    "bce-embedding-base_v1":"D:\\JHQ\\modelspace\\bce-embedding-base_v1"
}

EMBEDDING_MODEL = "bce-embedding-base_v1"
embeddings = HuggingFaceEmbeddings(model_name=embedding_model_dict[EMBEDDING_MODEL])

from langchain.vectorstores import Chroma
db = Chroma.from_documents(split_docs, embeddings,persist_directory='./persist') 

db.persist()  # save the database to disk

query = "RAG的核心思想是什么"
db = Chroma(persist_directory = './persist',embedding_function=embeddings)  # load the database from disk

similarDocs = db.similarity_search(query, k=3)

info = ""
for doc in similarDocs:
    print("----")
    if doc.page_content in info:
        continue
    print(doc.page_content)
    info = info + doc.page_content