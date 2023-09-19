import streamlit as st
from PyPDF2 import PdfReader
import langchain
from langchain.callbacks import StreamlitCallbackHandler
from langchain.chat_models import ChatOpenAI, ChatGooglePalm
from langchain.chains import ConversationalRetrievalChain
from langchain.llms import GooglePalm, OpenAI
from langchain.embeddings import GooglePalmEmbeddings, OpenAIEmbeddings, HuggingFaceEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain.chains.question_answering import load_qa_chain


# Set your Google API key
import os

import dotenv
from dotenv import load_dotenv

load_dotenv()

api_key1 = st.secrets["GOOGLE_API_KEY"]
#api_key2 = os.getenv("OPENAI_API_KEY")
os.environ["GOOGLE_API_KEY"] = api_key1
#os.environ["OPENAI_API_KEY"] = api_key2


def clear_submit():
    """
    Clear the Submit Button State
    Returns:

    """
    st.session_state["submit"] = False


# Streamlit UI
st.set_page_config(page_title="PDF ChatBot", page_icon="books")
st.title("Chat with your PDF")

# Define a caching function for PDF processing
@st.cache_resource()
def process_pdf(uploaded_file):
    # Read text from the uploaded PDF file
    pdfreader = PdfReader(uploaded_file)
    raw_text = ''.join(page.extract_text() for page in pdfreader.pages if page.extract_text())

    # Split the text using Character Text Splitter
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1500,
        chunk_overlap=200,
        length_function=len,
    )
    texts = text_splitter.split_text(raw_text)

    # Download embeddings from GooglePalm
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    #embeddings = GooglePalmEmbeddings()
    #embeddings = OpenAIEmbeddings()

    # Create a FAISS index from texts and embeddings
    document_search = FAISS.from_texts(texts, embeddings)

    return document_search



with st.sidebar:
    uploaded_file =  st.file_uploader("Upload your file",
    help="Only PDFs are Supported",
    on_change=clear_submit,
    type=['pdf'])

with st.sidebar:
    if not uploaded_file:
        st.warning("Upload your PDF to start chatting!")


if "messages" not in st.session_state or st.sidebar.button("Clear conversation history"):
    st.session_state["messages"] = [{"role": "assistant", "content": "How can I help you?"}]

if uploaded_file is not None:
    
    document_search = process_pdf(uploaded_file)

    for msg in st.session_state.messages:
        st.chat_message(msg["role"]).write(msg["content"])
    
    if prompt := st.chat_input(placeholder="Ask?"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        st.chat_message("user").write(prompt)
    
        memory = ConversationBufferMemory(memory_key="chat_history", input_key="question", human_prefix= "", ai_prefix= "")

        for i in range(0, len(st.session_state.messages), 2):
            if i + 1 < len(st.session_state.messages):
                current_message = st.session_state.messages[i]
                next_message = st.session_state.messages[i + 1]
                
                current_role = current_message["role"]
                current_content = current_message["content"]
                
                next_role = next_message["role"]
                next_content = next_message["content"]
                
                # Concatenate role and content for context and output
                context = f"{current_role}: {current_content}\n{next_role}-said: {next_content}"
                
                memory.save_context({"question": context}, {"output": ""})


            
        prompt_template = "You are a helpful assistant.\
                chat humbly. Answer my question from the provided context.\
                Use the following pieces of context to answer the question at the end. Your answer should be less than 30 words.\
                If you don't know the answer, just say that you don't know.\
                this is the context: {context}\
                This is chat history between you and user: {chat_history}\
                Question: {question}\
                Answer: "


        PROMPT = PromptTemplate(
            template=prompt_template, input_variables=["context", "question", "chat_history"]
        )

         # Run the question-answering chain
        docs = document_search.similarity_search(prompt, kwargs=3)

            # Load question-answering chain
        chain = load_qa_chain(GooglePalm(temperature=0.9), verbose= True, prompt = PROMPT,memory=memory, chain_type="stuff")
            
        #chain = load_qa_chain(ChatOpenAI(temperature=0.9, model="gpt-3.5-turbo-0613", streaming=True) , verbose= True, prompt = PROMPT, memory=memory,chain_type="stuff")


        with st.chat_message("assistant"):
            st_cb = StreamlitCallbackHandler(st.container(), expand_new_thoughts=True)
        
            response = chain.run(input_documents=docs, question = prompt, callbacks=[st_cb])
            st.session_state.messages.append({"role": "assistant", "content": response})
            
            st.write(response)





    

