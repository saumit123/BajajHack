import os
import re
from dotenv import load_dotenv

import pinecone
from langchain_pinecone import Pinecone as LangChainPinecone

from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.document_loaders import PyPDFLoader
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document

from langchain_experimental.text_splitter import SemanticChunker

load_dotenv()
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENVIRONMENT = os.getenv("PINECONE_ENVIRONMENT")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

if not all([PINECONE_API_KEY, PINECONE_ENVIRONMENT, GOOGLE_API_KEY]):
    raise ValueError("API keys for Pinecone and Google are required. Please check your .env file.")

def extract_and_process_pdf(pdf_path: str) -> list[Document]:
    print(f"Loading and extracting text from '{pdf_path}'...")
    loader = PyPDFLoader(pdf_path)
    pages = loader.load()

    cleaned_pages = []
    for page in pages:
        cleaned_text = re.sub(r"(\w+)-\n(\w+)", r"\1\2", page.page_content)
        cleaned_text = re.sub(r"(?<!\n\s)\n(?!\s\n)", " ", cleaned_text.strip())
        cleaned_text = re.sub(r"\n\s*\n", "\n\n", cleaned_text)

        cleaned_pages.append(Document(
            page_content=cleaned_text,
            metadata=page.metadata
        ))

    print("Text extraction and cleaning complete.")
    return cleaned_pages

def create_rag_chain(pdf_path: str):
    index_name = "insurance-hackathon"

    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    pinecone_client = pinecone.Pinecone(api_key=PINECONE_API_KEY, environment=PINECONE_ENVIRONMENT)

    if index_name not in pinecone_client.list_indexes().names():
        print(f"Creating new Pinecone index: {index_name}")
        pinecone_client.create_index(
            name=index_name,
            dimension=768,
            metric='cosine',
            spec=pinecone.ServerlessSpec(cloud='aws', region='us-east-1')
        )

        documents = extract_and_process_pdf(pdf_path)

        print("Chunking document with Semantic Chunker...")
        text_splitter = SemanticChunker(embeddings)
        docs = text_splitter.split_documents(documents)

        print(f"Uploading {len(docs)} document chunks to Pinecone...")
        vector_store = LangChainPinecone.from_documents(docs, embeddings, index_name=index_name)
        print("Ingestion complete.")
    else:
        print(f"Connecting to existing Pinecone index: {index_name}")
        vector_store = LangChainPinecone.from_existing_index(index_name=index_name, embedding=embeddings)

    retriever = vector_store.as_retriever()
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash-latest", temperature=0)
    prompt = ChatPromptTemplate.from_template("""
You are an expert insurance policy analyst. Your job is to give a clear and concise answer to a question about the policy.
Based *only* on the provided context, answer the user's question with a single line starting with "Yes,", "No,", or "Uncertain,".

<context>
{context}
</context>

Question: {input}
""")

    document_chain = create_stuff_documents_chain(llm, prompt)
    return create_retrieval_chain(retriever, document_chain)

if __name__ == "__main__":
    PDF_FILE_PATH = "policy.pdf"
    print("Initializing the Insurance Policy RAG system with Pinecone and PyPDF Extractor...")
    rag_chain = create_rag_chain(pdf_path=PDF_FILE_PATH)
    print("\n✅ System ready. Ask your questions about the insurance policy.")

    while True:
        question = input("\n> ")
        if question.lower() in ["exit", "quit"]:
            break

        response = rag_chain.invoke({"input": question})
        print(f"Answer: {response['answer']}")
