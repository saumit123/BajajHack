import os
from dotenv import load_dotenv

from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain

def main():
    """
    Main function to set up the RAG chain and handle user queries.
    """
    # Load environment variables from .env file
    load_dotenv()
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("GOOGLE_API_KEY not found. Please set it in your .env file.")

    ### --- ONE-TIME SETUP --- ###
    print("Step 1: Loading and Splitting PDF...")
    loader = PyPDFLoader("dummy_insurance_policy.pdf")
    docs = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    split_documents = text_splitter.split_documents(docs)

    print("Step 2: Creating embeddings and storing in FAISS...")
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_documents(split_documents, embeddings)

    print("Step 3: Defining LLM and Prompt...")
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash-latest", temperature=0)
    prompt = ChatPromptTemplate.from_template("""
    Answer the following question based only on the provided context.
    If you don't know the answer, just say that you don't know. Do not make up an answer.

    <context>
    {context}
    </context>

    Question: {input}
    """)

    print("Step 4: Creating the RAG Chain...")
    document_chain = create_stuff_documents_chain(llm, prompt)
    retriever = vector_store.as_retriever()
    retrieval_chain = create_retrieval_chain(retriever, document_chain)
    
    print("\n--- Setup Complete. Ready for questions. ---")

    ### --- INTERACTIVE QUERY LOOP --- ###
    while True:
        # Get user input from the command line
        user_question = input("\nAsk a question about your policy (or type 'exit' to quit): ")

        # Check if the user wants to exit
        if user_question.lower() in ["exit", "quit"]:
            print("Exiting... Goodbye! 👋")
            break
        
        # If not exiting, invoke the chain with the user's question
        response = retrieval_chain.invoke({"input": user_question})
        
        # Print the answer
        print(f"\nAnswer: {response['answer']}")


if __name__ == "__main__":
    main()                                                                                