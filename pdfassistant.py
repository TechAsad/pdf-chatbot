import streamlit as st
from PyPDF2 import PdfReader
import langchain
from textwrap import dedent
import pandas as pd
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.callbacks import StreamlitCallbackHandler
from langchain_openai import ChatOpenAI
from langchain_community.chat_models import ChatGooglePalm
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores.faiss import FAISS
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
import tempfile
from langchain.document_loaders.csv_loader import CSVLoader
from langchain.document_loaders.pdf import PyPDFLoader
from langchain.document_loaders.word_document import UnstructuredWordDocumentLoader
from langchain.chains.question_answering import load_qa_chain
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.agents import load_tools
import os
from io import BytesIO
from langdetect import detect
from gtts import gTTS
from langchain.prompts import (
    ChatPromptTemplate
)




st.set_page_config(page_title='Personal Chatbot', page_icon='books')



st.markdown(
    """
    <style>
        [data-testid=stImage]{
            text-align: center;
            display: block;
            margin-left: 10%;
            margin-right:10%;
            width: 100%;
    }
    img {
        border-radius: 50%;
        align: center;
    }
    </style>
    """, unsafe_allow_html=True
)



st.image("tenlancer.png", width=80)

st.markdown("<h3 style='text-align: center; color: white;'> Knowledge Query Assistant </h3>", unsafe_allow_html=True)




st.markdown(
    """
    <style>
    [data-testid="stChatMessageContent"] p{
        font-size: 1.2rem;
        color: #404040
    }
    </style>
    """, unsafe_allow_html=True
)


google_api_key = st.secrets["GOOGLE_API_KEY"]
#api_key2 = st.secrets["OPENAI_API_KEY"]
os.environ["GOOGLE_API_KEY"] = google_api_key



st.sidebar.header("options")
st.sidebar.subheader("Please Choose the AI Engine")
use_google = st.sidebar.checkbox("Use Free AI", value =True)
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
    
    
st.sidebar.subheader('Created by Engr. Muhammad Asadullah')

# Adding links to social accounts
st.sidebar.markdown("[LinkedIn](https://www.linkedin.com/in/asad18/)")
st.sidebar.markdown("[GitHub](https://github.com/TechAsad)")
st.sidebar.markdown("[Fiverr](https://www.fiverr.com/promptengr?source=gig_page&gigs=slug%3Acreate-streamlit-and-gradio-web-apps-for-ai-and-data-analysis%2Cpckg_id%3A1&is_choice=true)")
st.sidebar.markdown("[Website](https://tenlancer.com/)")
########--Save PDF--########
    

def text_to_audio(response, lang):
    audio_buffer = BytesIO()
    audio_file = gTTS(text=response, lang=lang, slow=False)
    audio_file.write_to_fp(audio_buffer)
    audio_buffer.seek(0)
    return audio_buffer


def main():
   # try:
        if (use_openai and openai_api_key) or use_google:
            if uploaded_file:
                db = processing_csv_pdf_docx(uploaded_file)
                for file in uploaded_file:
                    st.success(f'Your File: {file.name} is Embedded', icon="✅")
            
            for msg in st.session_state.messages:
                if msg["role"] == "user":
                    st.chat_message("user", avatar="user.png").write(msg["content"])
                
                if msg["role"] == "Assistant":
                    st.chat_message("Assistant", avatar="logo.png").write(msg["content"])
                    
                    st.audio(msg["audio_content"], format='audio/wav') 
                    #st.audio(audio_msg, format='audio/mp3').audio(audio_msg)
    
            
            if prompt := st.chat_input(placeholder="Type your question!"):
                st.session_state.messages.append({"role": "user", "content": prompt})
                st.chat_message("user", avatar="user.png").write(prompt)
                memory = ConversationBufferMemory(memory_key="chat_history", input_key="question", human_prefix= "User", ai_prefix= "Assistant")
                user_message = {"role": "user", "content": prompt}
                
                
                for i in range(0, len(st.session_state.messages), 2):
                    if i + 1 < len(st.session_state.messages):
                        user_prompt = st.session_state.messages[i]
                        ai_res = st.session_state.messages[i + 1]
                        
                        
                        current_content = user_prompt["content"]
                        
                        
                        next_content = ai_res["content"]
                        
                        # Concatenate role and content for context and output
                        user = f" {current_content}"
                        ai = f" {next_content}"
                        
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
                You are a helpful assistant.
                talk humbly. Answer the question from the provided context. Do not answer from your own training data.
                Use the following pieces of context to answer the question at the end.
                If you don't know the answer, just say that you don't know. Do not makeup any answer.
                Do not answer hypothetically. Do not answer in more than 100 words.
                Please Do Not say: "Based on the provided context"
                
                
                this is the context:
                ---------
                {context}
                ---------

                Current Conversation: 
                ---------
                {chat_history}
                ---------

                Question: 
                {question}

                Helpful Answer: 
                """)
                
                

                PROMPT = PromptTemplate(
                    template=prompt_template, input_variables=["context", "question", "chat_history"]
                )

                # Run the question-answering chain
                
                
                
                    # Load question-answering chain
                chain = load_qa_chain(llm=llm, verbose= True, prompt = PROMPT,memory=memory, chain_type="stuff")
                    
                #chain = load_qa_chain(ChatOpenAI(temperature=0.9, model="gpt-3.5-turbo-0613", streaming=True) , verbose= True, prompt = PROMPT, memory=memory,chain_type="stuff")

                with st.chat_message("Assistant",  avatar="logo.png"):
                    st_cb = StreamlitCallbackHandler(st.container())
                    if prompt.lower() in greetings:
                        response = 'Hi, how are you? I am here to help you get information from your file. How can I assist you?'
                        
                        
                        lang = "en"
                            
                            
                            
                        audio_buffer = text_to_audio(response, lang)
                        #st.audio(audio_buffer, format='audio/mp3')
                        st.session_state.messages.append({"role": "Assistant", "content": response, "audio_content": audio_buffer})
                        
                    elif prompt.lower() in compliment:
                        response = 'My pleasure! If you have any more questions, feel free to ask.'
                        
                        
                        lang = "en"
                            
                            
                            
                        audio_buffer = text_to_audio(response, lang)
                        #st.audio(audio_buffer, format='audio/mp3')
                        st.session_state.messages.append({"role": "Assistant", "content": response, "audio_content": audio_buffer})
                        
                    elif uploaded_file:
                        with st.spinner('Bot is typing ...'):
                            docs = db.similarity_search(prompt, k=5, fetch_k=10)
                            response = chain.run(input_documents=docs, question=prompt)
                            
                            
                            lang = detect(response)
                            
                            
                            
                            audio_buffer = text_to_audio(response, lang)
                           # st.audio(audio_buffer, format='audio/mp3')
                            #st.session_state.audio.append({"role": "Assistant", "audio": audio_buffer})
                            st.session_state.messages.append({"role": "Assistant", "content": response, "audio_content": audio_buffer})
                           
                            assistant_message = {"role": "assistant", "content": response}
                    else:
                        with st.spinner('Bot is typing ...'):
                            prompt_chat = ChatPromptTemplate.from_template("you are a helpful assistant, Answer the question with your knowledge.\n\n current conversation: {chat_history} \n\n Question: {question} \n\n Answer:")
                            chain = prompt_chat | llm
                            response = chain.invoke({"chat_history": memory, "question": prompt}).content
                            
                            
                            lang = detect(response)
                            
                            
                            
                            audio_buffer = text_to_audio(response, lang)
                            #st.audio(audio_buffer, format='audio/mp3')
                            #st.session_state.audio.append({"role": "Assistant", "audio": audio_buffer})
                            st.session_state.messages.append({"role": "Assistant", "content": response, "audio_content": audio_buffer})
                            
                            assistant_message = {"role": "assistant", "content": response}
                                            
                    st.write(response)             
                st.audio(audio_buffer, format='audio/wav')                        
                    
                            
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




