import streamlit as st
from PyPDF2 import PdfReader
import langchain
from langchain.callbacks import StreamlitCallbackHandler
from langchain.chat_models import ChatOpenAI, ChatGooglePalm
from langchain.chains import ConversationalRetrievalChain, LLMChain
from langchain.llms import GooglePalm, OpenAI
from langchain.embeddings import GooglePalmEmbeddings, OpenAIEmbeddings, HuggingFaceEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.prompts import PromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate, ChatPromptTemplate
from langchain.memory import ConversationBufferMemory
import tempfile
from langchain.document_loaders.csv_loader import CSVLoader
from langchain.document_loaders.pdf import PyPDFLoader
from langchain.document_loaders.word_document import UnstructuredWordDocumentLoader
from langchain.chains.question_answering import load_qa_chain

import os



import base64
def add_bg_from_local(image_file):
    with open(image_file, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read())
    
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url(data:image/png;base64,{encoded_string.decode()});
            background-size: cover;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )
    
st.set_page_config(
    page_title="Pdf/Docx/Csv Chat Bot",
    page_icon=":books:",
)

add_bg_from_local('bg2.jpg') 
 

def sidebar_bg(side_bg):

   side_bg_ext = 'png'

   st.markdown(
      f"""
      <style>
      [data-testid="stSidebar"] > div:first-child {{
          background: url(data:image/{side_bg_ext};base64,{base64.b64encode(open(side_bg, "rb").read()).decode()});
      }}
      </style>
      """,
      unsafe_allow_html=True,
      )
   
sidebar_bg("bg.png")


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
    return "If there is nothing below, just Say: 'Choose Right Side Bot for Csv'\n"




# Streamlit UI
st.set_page_config(page_title="PDF ChatBot", page_icon="books")
st.title("Chat with your PDF")

# Define a caching function for PDF processing
@st.cache_resource()
def processing_pdf_docx(uploaded_file):
    # Read text from the uploaded PDF file

    raw_text = '->\n'
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
        chunk_size=2000,
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
    uploaded_file =  st.file_uploader("Upload your files",
    help="Multiple Files are Supported",
    on_change=clear_submit,
    type=['pdf', 'docx', 'csv'], accept_multiple_files= True)

with st.sidebar:
    if not uploaded_file:
        st.warning("Upload your PDF to start chatting!")


if 'history' not in st.session_state:  
                st.session_state['history'] = []

if "messages" not in st.session_state or st.sidebar.button("Clear conversation history"):
    st.session_state["messages"] = [{"role": "assistant", "content": "How can I help you?"}]

if uploaded_file is not None:
    
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
                context = f"{current_role}-replied: {current_content}\n{next_role}-asked: {next_content}"
                
                memory.save_context({"question": context}, {"output": ""})


                    
                prompt_template = r"""
    -You are a helpful assistant.
    -talk humbly. Answer my question from the provided context.
    -Use the following pieces of context to answer the question at the end. Your answer should be less than 30 words.
    -If you don't know the answer, just say that you don't know.
    -this is the context:
    ---------
    {context}
    ---------
    
    This is chat history between you and user: 
    ---------
    {chat_history}
    ---------
    
    New Question: {question}
    
    Answer: 
    """


                PROMPT = PromptTemplate(
                    template=prompt_template, input_variables=["context", "question", "chat_history"]
                )

                # Run the question-answering chain
                docs = document_search.similarity_search(prompt, kwargs=3)

                    # Load question-answering chain
                chain = load_qa_chain(llm=llm, verbose= True, prompt = PROMPT,memory=memory, chain_type="stuff")
                    
                #chain = load_qa_chain(ChatOpenAI(temperature=0.9, model="gpt-3.5-turbo-0613", streaming=True) , verbose= True, prompt = PROMPT, memory=memory,chain_type="stuff")

        with st.chat_message("assistant"):
            st_cb = StreamlitCallbackHandler(st.container(), expand_new_thoughts=True)
        
            response = chain.run(input_documents=docs, question = prompt, callbacks=[st_cb])
            st.session_state.messages.append({"role": "assistant", "content": response})
            
            st.write(response)





    

