import os
from dotenv import load_dotenv

from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain

# Load environment variables from .env file
load_dotenv()
# Ensure the GOOGLE_API_KEY is available
api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    raise ValueError("GOOGLE_API_KEY not found. Please set it in your .env file.")

### 1. Load and Split the Document
print("Step 1: Loading and Splitting PDF...")
loader = PyPDFLoader(r"C:\Users\pradh\Downloads\ICICI_IPru_iProtect_Supreme.pdf")
docs = loader.load()
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
split_documents = text_splitter.split_documents(docs)
print(f"   -> Split into {len(split_documents)} chunks.")


### 2. Create Embeddings and Store in FAISS
print("Step 2: Creating embeddings and storing in FAISS...")
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
vector_store = FAISS.from_documents(split_documents, embeddings)
print("   -> FAISS vector store created successfully.")


### 3. Define the LLM and Prompt
print("Step 3: Defining LLM and Prompt...")
# Use the fast and efficient Flash model
llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0)

# The prompt instructs the AI to answer based *only* on the provided context
prompt = ChatPromptTemplate.from_template("""
Answer the following question based only on the provided context.
If you don't know the answer, just say that you don't know. Do not make up an answer.

<context>
{context}
</context>

Question: {input}
""")
print("   -> Components defined.")


### 4. Create the RAG Chain
print("Step 4: Creating the RAG Chain...")
# This chain will take a question, retrieve relevant documents, and pass them to the LLM
document_chain = create_stuff_documents_chain(llm, prompt)

# Create a retriever from our vector store
retriever = vector_store.as_retriever()

# The final retrieval chain that ties everything together
retrieval_chain = create_retrieval_chain(retriever, document_chain)
print("   -> RAG chain created successfully.")


### 5. Ask a Question
print("\n--- Ready to answer questions ---")
question = "What is the coverage for water damage and what is excluded?"

response = retrieval_chain.invoke({"input": question})

print(f"\nQuestion: {question}")
print(f"\nAnswer: {response['answer']}")

# You can also inspect the retrieved documents
# print("\n--- Retrieved Context ---")
# for i, doc in enumerate(response["context"]):
#     print(f"Context {i+1}:\n{doc.page_content}\n")