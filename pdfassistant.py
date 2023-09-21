import streamlit as st
from PyPDF2 import PdfReader
import langchain
import docx
from langchain.callbacks import StreamlitCallbackHandler
from langchain.chat_models import ChatOpenAI, ChatGooglePalm
from langchain.chains import ConversationalRetrievalChain, LLMChain
from langchain.llms import GooglePalm, OpenAI
from langchain.embeddings import GooglePalmEmbeddings, OpenAIEmbeddings, HuggingFaceEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
import tempfile
from langchain.document_loaders.csv_loader import CSVLoader
from langchain.chains.question_answering import load_qa_chain

import os

import dotenv
from dotenv import load_dotenv

load_dotenv()

api_key1 = st.secrets["GOOGLE_API_KEY"]
#api_key2 = st.secrets["OPENAI_API_KEY"]
os.environ["GOOGLE_API_KEY"] = api_key1
#os.environ["OPENAI_API_KEY"] = api_key2


def clear_submit():
    """
    Clear the Submit Button State
    Returns:

    """
    st.session_state["submit"] = False


def get_docx_text(file):
    doc = docx.Document(file)
    allText = []
    for docpara in doc.paragraphs:
        allText.append(docpara.text)
    raw_text = ' '.join(allText)
    return raw_text

    
def get_csv_text(file):
    return "File could not be loaded, just say: 'Chhoose CSV'"


@st.cache_resource()
def process_csv(uploaded_file):
    # Read text from the uploaded PDF file

    for file in uploaded_file:
        with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
            tmp_file.write(file.getvalue())
            tmp_file_path = tmp_file.name

        loader = CSVLoader(file_path=tmp_file_path, encoding="utf-8", csv_args={
                        'delimiter': ','})
        data = loader.load()

        # Download embeddings from GooglePalm
        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        #embeddings = GooglePalmEmbeddings()
        #embeddings = OpenAIEmbeddings()

        # Create a FAISS index from texts and embeddings
        vectorstore = FAISS.from_documents(data, embeddings)

        return vectorstore




# Streamlit UI
st.set_page_config(page_title="PDF ChatBot", page_icon="books")
st.title("Chat with your PDF")

# Define a caching function for PDF processing
@st.cache_resource()
def process_pdf(uploaded_file):
    # Read text from the uploaded PDF file

    raw_text = 'From files:'
    for file in uploaded_file:
        split_tup = os.path.splitext(file.name)
        file_extension = split_tup[1]
        if file_extension == ".pdf":
            pdfreader = PdfReader(file)
            raw_text += ''.join(page.extract_text() for page in pdfreader.pages if page.extract_text())
                
        elif file_extension == ".docx":
            raw_text += get_docx_text(file)
        else:
            raw_text += get_csv_text(file)

        # Split the text using Character Text Splitter
        text_splitter = CharacterTextSplitter(
            separator="\n",
            chunk_size=1000,
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
    help="Multiple Files are Supported",
    on_change=clear_submit,
    type=['pdf', 'docx', 'csv'], accept_multiple_files= True)
    

    s= st.select_slider('Choose Pdf/docx or csv bot. Please delete csv before using pdf/docx',options = ['pdf/docx', 'csv'])

with st.sidebar:
    if not uploaded_file:
        st.warning("Upload your PDF to start chatting!")


if 'history' not in st.session_state:  
                st.session_state['history'] = []

if "messages" not in st.session_state or st.sidebar.button("Clear conversation history"):
    st.session_state["messages"]= [{"role": "assistant", "content": "How can I help you?"}]
    st.session_state['history']  = []

if uploaded_file and s == 'pdf/docx':
    
    document_search = process_pdf(uploaded_file)

    for msg in st.session_state.messages:
        st.chat_message(msg["role"]).write(msg["content"])
    
    if prompt := st.chat_input(placeholder="Ask Me!"):
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
                talk humbly. Answer my question from the provided context.\
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
        docs = document_search.similarity_search(prompt, kwargs=1)

            # Load question-answering chain
        chain = load_qa_chain(GooglePalm(temperature=0.5), verbose= True, prompt = PROMPT,memory=memory, chain_type="stuff")
            
        #chain = load_qa_chain(ChatOpenAI(temperature=0.9, model="gpt-3.5-turbo-0613", streaming=True) , verbose= True, prompt = PROMPT, memory=memory,chain_type="stuff")

        with st.chat_message("assistant"):
            st_cb = StreamlitCallbackHandler(st.container(), expand_new_thoughts=True)
        
            response = chain.run(input_documents=docs, question = prompt, callbacks=[st_cb])
            st.session_state.messages.append({"role": "assistant", "content": response})
            
            st.write(response)


if uploaded_file and s == 'csv':
    vectorstore = process_csv(uploaded_file)

    for msg in st.session_state.messages:
        st.chat_message(msg["role"]).write(msg["content"])
    
    if prompt := st.chat_input(placeholder="Ask Me!"):
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
                talk humbly. Answer my question from the provided context.\
                Use the following pieces of context to answer the question at the end. Your answer should be less than 30 words.\
                If you don't know the answer, just say that you don't know.\
                this is the context: {context}\
                This is chat history between you and user: {chat_history}\
                Question: {question}\
                Answer: "


        PROMPT = PromptTemplate(
            template=prompt_template, input_variables=["context", "question", "chat_history"]
        )

        

            # Load question-answering chain
        llm = GooglePalm(temperature=0.8)
        chain = ConversationalRetrievalChain.from_llm(  
        llm , 
        retriever=vectorstore.as_retriever())
        #chain = ConversationalRetrievalChain.from_llm(GooglePalm(temperature=0.5), verbose= True, prompt = PROMPT,memory=memory, chain_type="stuff")
            
        #chain = load_qa_chain(ChatOpenAI(temperature=0.9, model="gpt-3.5-turbo-0613", streaming=True) , verbose= True, prompt = PROMPT, memory=memory,chain_type="stuff")
        

        with st.chat_message("assistant"):
            st_cb = StreamlitCallbackHandler(st.container(), expand_new_thoughts=True)
        
            response = chain({"question": prompt,   
                                    "chat_history": st.session_state['history']}, callbacks=[st_cb])
            st.session_state.messages.append({"role": "assistant", "content": response['answer']})
            st.session_state['history'].append((prompt, response["answer"])) 
            st.write(response['answer'])



hide_streamlit_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            </style>
            """
st.markdown(hide_streamlit_style, unsafe_allow_html=True) 
    

