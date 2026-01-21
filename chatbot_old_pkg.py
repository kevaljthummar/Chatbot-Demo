import streamlit as st
from PyPDF2 import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain_community.chains import LLMChain
from langchain_core.prompts import PromptTemplate

OPENAI_API_KEY = "sk-proj-cGUzIl60T4VY"

# Upload PDF files
st.header("My first Chatbot")


with st.sidebar:
    st.title("Your Documents")
    file = st.file_uploader(
        "Upload a PDF file and start asking questions", type="pdf")

# Extract the text
if file is not None:
    pdf_reader = PdfReader(file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()
        # st.write(text)

    # Break it into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        separators="\n",
        chunk_size=1000,
        chunk_overlap=150,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    # st.write(chunks)

    # Genrating embeddings
    embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)

    # Creating vector store - FAISS (Facebook AI Similarity Search)
    # embedding (OpenAI)
    # Initizling FAISS
    # Store_Chunks & embeddings
    vector_store = FAISS.from_texts(chunks, embeddings)

    # Get user questions
    user_question = st.text_input("Type Your question here")

    # Do something search
    if user_question:
        match = vector_store.similarity_search(user_question)
        st.write(match)

        prompt = PromptTemplate(
            input_variables=["question"],
            template="Answer clearly: {question}"
        )

        # define the LLM
        llm = ChatOpenAI(
            openai_api_key=OPENAI_API_KEY,
            temperature=0,
            max_tokens=1000,
            model_name="gpt-3.5-turbo"
        )

        # output results
        # chain -> take the question, get relevant document, pass it to the LLM, generate the output
        chain = LLMChain(llm=llm, prompt=prompt)
        response = chain.run(input_documents=match, question=user_question)
        st.write(response)
