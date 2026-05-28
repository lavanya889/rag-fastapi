from fastapi import UploadFile,File,APIRouter 
import os 
import shutil 
from app.services.pdf_services import extract_text_from_pdf,chunk_text,embedding_generate,embeddings_store,search_similar_chunks
from app.models.schemas import QuestionRequest
from sentence_transformers import SentenceTransformer
from app.services.llm_service import generate_answer

router=APIRouter() 

upload_dir="uploads" 

# Create uploads folder if not exists
os.makedirs(upload_dir, exist_ok=True)
 

@router.post('/upload_pdf') 
async def upload_pdf(file: UploadFile=File(...)):

    #validate pdf 

    if not file.filename.endswith(".pdf"):
        return {"only pdf files are accepted"} 
    
    # file_path
    
    file_path = os.path.join(upload_dir, file.filename)

    # Save file
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    

    extracted_text=extract_text_from_pdf(file_path)
    chunks=chunk_text(extracted_text)
    embedding_vectors=embedding_generate(chunks)
    # storing in chromadb 
    embeddings_store(chunks,embedding_vectors)
    return {
        "message": "PDF uploaded successfully",
        "filename": file.filename,
        "total_chunks": len(chunks),
        "status": "Embeddings stored in ChromaDB"
        
    }

    # question asking api 

embedding_model=SentenceTransformer("all-MiniLM-L6-v2")
@router.post("/ask")
async def ask_question(request: QuestionRequest):
    question = request.question
    question_embedding = embedding_model.encode(question)
    results = search_similar_chunks(
        question_embedding
        )
    
    retrieved_chunks=results["documents"][0]

    # Combine chunks as context
    
    context = "\n".join(retrieved_chunks)

    answer=generate_answer(question,context)
    
    return{
        "question":question,
        "answer":answer,
        "retrieved_chunks":retrieved_chunks

    }