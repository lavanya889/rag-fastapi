from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter 
from sentence_transformers import SentenceTransformer
import chromadb
 
def extract_text_from_pdf(pdf_path:str):

    reader=PdfReader(pdf_path) 

    full_text="" 

    #loop through each page 
    for page in reader.pages:

        text=page.extract_text() 

        if text:
            full_text+= text+'\n'
    return full_text


#chunking text 

def chunk_text(text:str):
    text_splitter=RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )
    chunk=text_splitter.split_text(text) 
    return chunk

# converting text into embeddings 
embedding_model=SentenceTransformer("all-MiniLM-L6-v2") 

def embedding_generate(chunks):
    embeddings=embedding_model.encode(chunks)
    return embeddings 

#storing embeddings and chunks in chromadb

# create chroma client 
client =chromadb.Client() 
#creating collection 
collection=client.get_or_create_collection(name="pdf_collection")

def embeddings_store(chunk,embedding):
    for index,(chunks,embeddings) in enumerate(zip(chunk,embedding)):
        collection.add(
            ids=[str(index)],
            embeddings=[embeddings.tolist()],
            documents=[chunks]

        )
    return "embeddings stored successfully"


#similarity checking between vdb and user query

def search_similar_chunks(query_embedding):

    results = collection.query(
        query_embeddings=[query_embedding.tolist()],
        n_results=3
    )

    return results


