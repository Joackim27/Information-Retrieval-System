# src/helper.py
import os
import torch
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, pipeline
from langchain_core.language_models import LLM
from typing import Any, List, Optional

# Load environment variables
load_dotenv()
HF_API_KEY = os.getenv("HUGGINGFACEHUB_API_TOKEN", None)

# Model configuration
MODEL_ID = "google/flan-t5-large"

# Load model and tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForSeq2SeqLM.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    device_map="auto"
)

# Text generation pipeline
generator = pipeline(
    "text2text-generation",
    model=model,
    tokenizer=tokenizer
)

class LocalLLM(LLM):
    """Custom LLM wrapper for HuggingFace pipelines."""
    
    def _call(self, prompt: str, stop: Optional[List[str]] = None, **kwargs) -> str:
        # Generate response using the pipeline
        response = generator(
            prompt,
            max_new_tokens=1000,
            #temperature=0.1,
            # top_k=10,
            #top_p=0.9,
            do_sample=True 
        )
        return response[0]["generated_text"]
    
    @property
    def _llm_type(self) -> str:
        return "flan-t5-local"

def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text() or ""
    return text

def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=20
    )
    return text_splitter.split_text(text)

def get_vector_store(text_chunks):
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return FAISS.from_texts(text_chunks, embedding=embeddings)

def get_conversational_chain(vector_store):
    llm = LocalLLM()
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    
    return ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vector_store.as_retriever(),
        memory=memory
    )