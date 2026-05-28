import os 
from dotenv import load_dotenv 
load_dotenv()
from langchain_openai import ChatOpenAI 



 
# initializing openai model 
llm=ChatOpenAI(
    model="gpt-4o-mini",
    api_key=os.getenv("OPENAI_API_KEY"),
    temperature=0
)

def generate_answer(question,context):
    prompt = f""" You are an AI assistant.

    Answer the question ONLY from the provided context.

    If answer is not available in context,
    say:
    "Answer not found in the document."

    Context:
    {context}

    Question:
    {question}

    Answer:
    """
    response=llm.invoke(prompt)
    return response.content
