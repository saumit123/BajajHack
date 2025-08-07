from langchain_community.document_loaders import TextLoader,WebBaseLoader,PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import bs4
from langchain_google_genai import ChatGoogleGenerativeAI,GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv
load_dotenv()

#TextLoader

# loader = TextLoader(r'C:\Users\pradh\Desktop\Langchain Models\RAG\speech.txt')
# text_documents = loader.load()

llm = ChatGoogleGenerativeAI(
    model='gemini-1.5-flash',
    temperature=0.5,
    max_tokens=512)

#Load,chunk,index the HTML page

# loader=WebBaseLoader(web_paths=("https://lilianweng.github.io/posts/2023-06-23-agent/",),
#                      bs_kwargs=dict(parse_only=bs4.SoupStrainer(
#                          class_=("post-title","post-content","post-header")
#                      )))
# text_documents = loader.load()

#PDF Loader

loader = PyPDFLoader(r"C:\Users\pradh\Downloads\ICICI_IPru_iProtect_Supreme.pdf")
docs = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=200)
documents = text_splitter.split_documents(docs)

