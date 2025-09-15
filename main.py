import os
from dotenv import load_dotenv
from langchain_community.document_loaders import JSONLoader
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import CharacterTextSplitter
from langchain.chains.retrieval import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain import hub
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI


# Load environment variables from .env file
load_dotenv(dotenv_path="D:/RAG/.env", override=True)
api_key = os.getenv("GEMINI_API_KEY")

pdf_path = "./sample_input.json"

loader = JSONLoader(
    file_path=pdf_path,
    jq_schema=".",
    text_content=False,
)  # converting pdf to readable format


documents = loader.load()  # loader reads pdf pages and stores in test list

text_splitter = CharacterTextSplitter(
    chunk_size=1000,  # 1000 chars
    chunk_overlap=30,
    separator="\n",
)
split_documents = text_splitter.split_documents(documents)

embeddings = GoogleGenerativeAIEmbeddings(
    model="models/text-embedding-004",
    google_api_key=api_key,
)
vectorstore = FAISS.from_documents(split_documents, embeddings)

# Save the vector store
vectorstore.save_local("faiss_index")

# Load the vector store
new_vectorstore = FAISS.load_local(
    "faiss_index", embeddings, allow_dangerous_deserialization=True
)

retrieval_qa_chat_prompt = hub.pull("langchain-ai/retrieval-qa-chat")

llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",
    temperature=0.7,
    api_key=api_key,
)

combine_docs_chain = create_stuff_documents_chain(
    llm,
    retrieval_qa_chat_prompt,
)

retrieval_chain = create_retrieval_chain(
    new_vectorstore.as_retriever(k=10), combine_docs_chain
)

chat_history = []

while True:
    user_input = input("You: ").strip()
    if user_input.lower() in ["exit", "quit"]:
        break
    chat_history.append({"role": "user", "content": user_input})

    res = retrieval_chain.invoke(
        {
            "input": user_input,
            "chat_history": chat_history,
        }
    )
    print(f"Assistant: {res['answer']}")

    chat_history.append({"role": "assistant", "content": res["answer"]})
    print(chat_history)
