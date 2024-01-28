import streamlit as st
from PyPDF2 import PdfReader
import langchain
import docx
import pypdf 
from textwrap import dedent
import pandas as pd
from langchain_community.callbacks import StreamlitCallbackHandler
from langchain_openai import ChatOpenAI
from langchain_community.chat_models import ChatGooglePalm
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import CharacterTextSplitter, RecursiveCharacterTextSplitter
from langchain.vectorstores.faiss import FAISS
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
import tempfile
from langchain.document_loaders.csv_loader import CSVLoader
from langchain.document_loaders.pdf import PyPDFLoader
from langchain.document_loaders.word_document import UnstructuredWordDocumentLoader
from langchain.chains.question_answering import load_qa_chain
from langchain_community.document_loaders import DirectoryLoader
from vectordbsearch_tools import VectorSearchTools
from langchain_community.llms.ctransformers import CTransformers
from langchain_community.llms.ollama import Ollama
from langchain.llms.llamacpp import LlamaCpp
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.agents import load_tools
import os


google_api_key = st.secrets["GOOGLE_API_KEY"]
#api_key2 = st.secrets["OPENAI_API_KEY"]
os.environ["GOOGLE_API_KEY"] = google_api_key

st.set_page_config(page_title='Personal Chatbot', page_icon='books')
st.header('Knowledge Query Assistant')
st.write("I'm here to help you get information from your file.")
st.sidebar.title('Options')


st.sidebar.subheader("Please Choose the AI Engine")
use_google = st.sidebar.checkbox("Use Free Google AI")
use_openai = st.sidebar.checkbox("Use OpenAI with your API Key")

openai_api_key = st.sidebar.text_input("Enter your OpenAI API Key:", type="password")

def choose_llm():
    try:
        if use_google and use_openai:
            st.sidebar.warning("Please choose only one AI engine.")
            st.warning("Please choose only one AI engine.")
        elif use_google:
            llm = ChatGooglePalm(temperature=0.1)
        elif use_openai:
            if not openai_api_key:
                st.sidebar.warning("Please provide your OpenAI API Key.")
                st.warning("Please provide your OpenAI API Key.")
            llm = ChatOpenAI(api_key=openai_api_key, temperature=0.1)
        return llm
    except Exception as e:
        " "
         
            
llm = choose_llm()

if llm:
    st.sidebar.success("AI Engine selected")
else:
    st.sidebar.warning("Please choose an AI engine.")

@st.cache_resource(show_spinner=False)
def processing_csv_pdf_docx(uploaded_file):
    with st.spinner(text="Embedding Your Files"):

        # Read text from the uploaded PDF file
        data = []
        for file in uploaded_file:
            split_tup = os.path.splitext(file.name)
            file_extension = split_tup[1]
        
            if file_extension == ".pdf":

                with tempfile.NamedTemporaryFile(delete=False) as tmp_file1:
                    tmp_file1.write(file.getvalue())
                    tmp_file_path1 = tmp_file1.name
                    loader = PyPDFLoader(file_path=tmp_file_path1)
                    documents = loader.load()
                    text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=50)
                    data += text_splitter.split_documents(documents)


            if file_extension == ".csv":
                
                with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
                    tmp_file.write(file.getvalue())
                    tmp_file_path = tmp_file.name

                    loader = CSVLoader(file_path=tmp_file_path, encoding="utf-8", csv_args={
                                'delimiter': ','})
                    documents = loader.load()
                    
                    text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=50)
        
                    data += text_splitter.split_documents(documents)
                    st.sidebar.header(f"Data-{file.name}")
                    data1 = pd.read_csv(tmp_file_path)
                    st.sidebar.dataframe(data1)
            
            if file_extension == ".docx":

                with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
                    tmp_file.write(file.getvalue())
                    tmp_file_path = tmp_file.name
                    loader = UnstructuredWordDocumentLoader(file_path=tmp_file_path)
                    documents = loader.load()
                    text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=50)

                    data += text_splitter.split_documents(documents)
                

                # Download embeddings from GooglePalm
                embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
                #embeddings = GooglePalmEmbeddings()
                #embeddings = OpenAIEmbeddings()

                # Create a FAISS index from texts and embeddings

                vectorstore = FAISS.from_documents(data, embeddings)
                #vectorstore.save_local("./faiss")
                return vectorstore



with st.sidebar:
    uploaded_file =  st.file_uploader("Upload your files",
    help="Multiple Files are Supported",
    type=['pdf', 'docx', 'csv'], accept_multiple_files= True)


if not uploaded_file:
    st.warning("Upload your file(s) to start chatting!")
    


if 'history' not in st.session_state:  
        st.session_state['history'] = []


if "messages" not in st.session_state or st.sidebar.button("Clear conversation history"):
    st.session_state["messages"]= []
    st.session_state['history']  = []


########--Save PDF--########
    
def load_files():
    for file in uploaded_file:
        with open(os.path.join('./uploaded_files', file.name), 'wb') as f:
            f.write(file.getbuffer())


def main():
   # try:
        if (use_openai and openai_api_key) or use_google:
            if uploaded_file:
                load_files()
            db = processing_csv_pdf_docx(uploaded_file)
            for file in uploaded_file:
                st.success(f'File Embedded: {file.name}', icon="✅")
        
            for msg in st.session_state.messages:
                st.chat_message(msg["role"]).write(msg["content"])      
            
            if prompt := st.chat_input(placeholder="Type your question!"):
                st.session_state.messages.append({"role": "user", "content": prompt})
                st.chat_message("user").write(prompt)
                memory = ConversationBufferMemory(memory_key="chat_history", input_key="question", human_prefix= "", ai_prefix= "")
                user_message = {"role": "user", "content": prompt}
                
                
                for i in range(0, len(st.session_state.messages), 2):
                    if i + 1 < len(st.session_state.messages):
                        user_prompt = st.session_state.messages[i]
                        ai_res = st.session_state.messages[i + 1]
                        
                        current_role = user_prompt["role"]
                        current_content = user_prompt["content"]
                        
                        next_role = ai_res["role"]
                        next_content = ai_res["content"]
                        
                        # Concatenate role and content for context and output
                        user = f"{current_role}: {current_content}"
                        ai = f"{next_role}: {next_content}"
                        
                        memory.save_context({"question": user}, {"output": ai})

                # Get user input -> Generate the answer
                greetings = ['Hey', 'Hello', 'hi', 'hello', 'hey', 'helloo', 'hellooo', 'g morning', 'gmorning', 'good morning', 'morning',
                            'good day', 'good afternoon', 'good evening', 'greetings', 'greeting', 'good to see you',
                            'its good seeing you', 'how are you', "how're you", 'how are you doing', "how ya doin'", 'how ya doin',
                            'how is everything', 'how is everything going', "how's everything going", 'how is you', "how's you",
                            'how are things', "how're things", 'how is it going', "how's it going", "how's it goin'", "how's it goin",
                            'how is life been treating you', "how's life been treating you", 'how have you been', "how've you been",
                            'what is up', "what's up", 'what is cracking', "what's cracking", 'what is good', "what's good",
                            'what is happening', "what's happening", 'what is new', "what's new", 'what is neww', "g’day", 'howdy']
                compliment = ['thank you', 'thanks', 'thanks a lot', 'thanks a bunch', 'great', 'ok', 'ok thanks', 'okay', 'great', 'awesome', 'nice']
                            
                prompt_template =dedent(r"""
                You are a helpful assistant to help user find information from his documents.
                talk humbly. Answer the question from the provided context. Do not answer from your own training data.
                Use the following pieces of context to answer the question at the end.
                If you don't know the answer, just say that you don't know. Do not makeup any answer.
                Do not answer hypothetically. Do not answer in more than 100 words.
                Please Do Not say: "Based on the provided context"
                Always use the context to find the answer.
                
                this is the context from study material:
                ---------
                {context}
                ---------

                Current Conversation: 
                ---------
                {chat_history}
                ---------

                Question: {question}

                Helpful Answer: 
                """)
                
                

                PROMPT = PromptTemplate(
                    template=prompt_template, input_variables=["context", "question", "chat_history"]
                )

                # Run the question-answering chain
                
                
                
                    # Load question-answering chain
                chain = load_qa_chain(llm=llm, verbose= True, prompt = PROMPT,memory=memory, chain_type="stuff")
                    
                #chain = load_qa_chain(ChatOpenAI(temperature=0.9, model="gpt-3.5-turbo-0613", streaming=True) , verbose= True, prompt = PROMPT, memory=memory,chain_type="stuff")

                with st.chat_message("assistant"):
                    st_cb = StreamlitCallbackHandler(st.container())
                    if prompt.lower() in greetings:
                        response = 'Hi, how are you? I am here to help you get information from your file. How can I assist you?'
                        st.session_state.messages.append({"role": "Assistant", "content": response})
                        st.write(response)
                    elif prompt.lower() in compliment:
                        response = 'My pleasure! If you have any more questions, feel free to ask.'
                        st.session_state.messages.append({"role": "Assistant", "content": response})
                        st.write(response)
                    else:
                        with st.spinner('Bot is typing ...'):
                            #docs = VectorSearchTools.dbsearch(prompt)
                            if uploaded_file:
                                docs = db.similarity_search(prompt, k=5, fetch_k= 10)
                            else:
                                docs = ['Document(page_content= "Context not provide, answer the question from your knowledge")']
                            response = chain.run(input_documents=docs, question = prompt)#, callbacks=[st_cb])
                            st.session_state.messages.append({"role": "Assistant", "content": response})
                            
                            assistant_message = {"role": "assistant", "content": response}
                        
                                            
                            st.write(response)
                            
    #except Exception as e:
     #   "Sorry, there was a problem. A corrupted file or;"
      #  if use_google:
       #     "Google PaLM AI only take English Data and Questions. Or the AI could not find the answer in your provided document."
        #elif use_openai:
         #   "Please check your OpenAI API key"
         


hide_streamlit_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            </style>
            """
st.markdown(hide_streamlit_style, unsafe_allow_html=True) 


if __name__ == '__main__':
    main()




