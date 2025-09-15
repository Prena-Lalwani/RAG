from dotenv import load_dotenv
from langchain import hub
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, OpenAI
from langchain.chains.retrieval import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.vectorstores import FAISS


# Load environment variables from .env file
load_dotenv(dotenv_path="D:/RAG/.env", override=True)

# api_key = os.getenv("OPENAI_API_KEY")
pdf_path = "./2005.11401v4.pdf"
# print("OPENAI_API_KEY:", api_key[:8])
loader = PyPDFLoader(file_path=pdf_path)
documents = loader.load()

text_splitter = CharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=30,
    separator="\n",
)
split_documents = text_splitter.split_documents(documents)

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
vectorstore = FAISS.from_documents(split_documents, embeddings)

# Save the vector store
vectorstore.save_local("faiss_index")

# Load the vector store
new_vectorstore = FAISS.load_local(
    "faiss_index_react", embeddings, allow_dangerous_deserialization=True
)

retrieval_qa_chat_prompt = hub.pull("langchain-ai/retrieval-qa-chat")

combine_docs_chain = create_stuff_documents_chain(
    OpenAI(),
    retrieval_qa_chat_prompt,
)

retrieval_chain = create_retrieval_chain(
    new_vectorstore.as_retriever(), combine_docs_chain
)

res = retrieval_chain.invoke(
    {"input": "Give me the gist of Retrieval-Augmented Generation (RAG) in 3 sentences"}
)
print(res["answer"])
