from transformers import pipeline  

# references  
# https://huggingface.co/docs/transformers/en/main_classes/pipelines  

# initialize a question-answering pipeline with a pre-trained model  
qa_pipeline = pipeline("question-answering", model="distilbert-base-uncased-distilled-squad")  

# define your context and question  
context = "Hugging Face is a technology company that provides open-source NLP libraries ..."  
question = "What does Hugging Face provide?"  

# let the pipeline find the best answer based on the context provided  
answer = qa_pipeline(question=question, context=context)  

print(f"Question: {question}")  
print(f"Answer: {answer['answer']}")  



# Importing Necessary Libraries
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import pipeline
import uvicorn

# Creating the FastAPI Application
app = FastAPI()

# Defining Data Models
class ChatRequest(BaseModel):
    question: str
    context: str

class ChatResponse(BaseModel):
    answer: str

# Initializing the Question-Answering Pipeline
qa_pipeline = pipeline("question-answering", model="distilbert-base-uncased-distilled-squad")

# Creating the /chat endpoint
@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    try:
        result = qa_pipeline(question=request.question, context=request.context)
        return ChatResponse(answer=result['answer'])
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Running the app server
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
