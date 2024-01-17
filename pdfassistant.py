import streamlit as st
from PyPDF2 import PdfReader
import langchain
import docx
import pypdf 
import pandas as pd
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



api_key1 = st.secrets["GOOGLE_API_KEY"]
#api_key2 = st.secrets["OPENAI_API_KEY"]
os.environ["GOOGLE_API_KEY"] = api_key1
#os.environ["OPENAI_API_KEY"] = api_key2
llm = ChatGooglePalm(temperature=0.3, max_output_tokens= 1024 ,verbose=True,streaming=True)
#llm = OpenAI(temperature=0.9,verbose=True)


def get_docx_text(file):
    doc = docx.Document(file)
    allText = []
    for docpara in doc.paragraphs:
        allText.append(docpara.text)
    raw_text = ' '.join(allText)
    return raw_text

    
def get_csv_text(file):
    return "Empty"


st.set_page_config(page_title="Document ChatBot", page_icon="books")
st.title("Chat with your Documents")

with st.sidebar:
    #option= st.select_slider('Choose the Small or Large data Bot. Left side only works with Pdf and Docx.' ,options = ['Small Size Pdf/Docx', 'Large Size Pdf/Dcx and Csv'])
    st.subheader("Please Choose Between")
    use_small= st.checkbox("Small Pdf/Docx <10MB")
    use_large= st.checkbox("Lrage Pdf/Docx and Csv")

@st.cache_resource(show_spinner=False)
def processing_pdf_docx(uploaded_file):
    with st.spinner(text="Getting Ready"):
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
            chunk_size=500,
            chunk_overlap=50,
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
    type=['pdf', 'docx', 'csv'], accept_multiple_files= True)

with st.sidebar:
    if not uploaded_file:
        st.warning("Upload your files to start chatting!")


if 'history' not in st.session_state:  
        st.session_state['history'] = []


if "messages" not in st.session_state or st.sidebar.button("Clear conversation history"):
    st.session_state["messages"]= []
    st.session_state['history']  = []


########--Main PDF--########

if use_small:# option == 'Small Size Pdf/Docx':
    use_large = False

        #s= st.select_slider('Choose Pdf/docx or csv bot. Please delete csv before using pdf/docx',options = ['pdf/docx', 'csv'])


    if uploaded_file is not None:
        
            
            document_search = processing_pdf_docx(uploaded_file)

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
                        context = f"{current_role}: {current_content}\n{next_role}: {next_content}"
                        
                        memory.save_context({"question": context}, {"output": ""})


                    
                prompt_template = r"""
    -You are a helpful assistant.
    -talk humbly. Answer my question from the provided context.
    -Use the following pieces of context to answer the question at the end.
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
                docs = document_search.similarity_search(prompt, k=3)

                    # Load question-answering chain
                chain = load_qa_chain(llm=llm, verbose= True, prompt = PROMPT,memory=memory, chain_type="stuff")
                    
                #chain = load_qa_chain(ChatOpenAI(temperature=0.9, model="gpt-3.5-turbo-0613", streaming=True) , verbose= True, prompt = PROMPT, memory=memory,chain_type="stuff")

                with st.chat_message("assistant"):
                    st_cb = StreamlitCallbackHandler(st.container(), expand_new_thoughts=True)
                
                    response = chain.run(input_documents=docs, question = prompt, callbacks=[st_cb])
                    st.session_state.messages.append({"role": "Assistant", "content": response})
                    
                    st.write(response)


@st.cache_resource(show_spinner=False)
def processing_csv_pdf_docx(uploaded_file):
    with st.spinner(text="Getting Ready"):

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
                    text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
                    data += text_splitter.split_documents(documents)


            if file_extension == ".csv":
                
                with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
                    tmp_file.write(file.getvalue())
                    tmp_file_path = tmp_file.name

                    loader = CSVLoader(file_path=tmp_file_path, encoding="utf-8", csv_args={
                                'delimiter': ','})
                    documents = loader.load()
                    
                    text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        
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
                    text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)

                    data += text_splitter.split_documents(documents)
                

        # Download embeddings from GooglePalm
        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        #embeddings = GooglePalmEmbeddings()
        #embeddings = OpenAIEmbeddings()

        # Create a FAISS index from texts and embeddings
        vectorstore = FAISS.from_documents(data, embeddings)

        return vectorstore


########--Main CSV--########


if use_large: #option == 'Large Size Pdf/Dcx and Csv':
    use_small = False
    
        #s= st.select_slider('Choose Pdf/docx or csv bot. Please delete csv before using pdf/docx',options = ['pdf/docx', 'csv'])


    if uploaded_file:
        
        vectorstore = processing_csv_pdf_docx(uploaded_file)

        for msg in st.session_state.messages:
            st.chat_message(msg["role"]).write(msg["content"])
        
        if prompt := st.chat_input(placeholder="Ask Me!"):
            st.session_state.messages.append({"role": "user", "content": prompt})
            st.chat_message("user").write(prompt)
        
            memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True, input_key="question", human_prefix= "", ai_prefix= "")

            for i in range(0, len(st.session_state.messages), 2):
                if i + 1 < len(st.session_state.messages):
                    current_message = st.session_state.messages[i]
                    next_message = st.session_state.messages[i + 1]
                    
                    current_role = current_message["role"]
                    current_content = current_message["content"]
                    
                    next_role = next_message["role"]
                    next_content = next_message["content"]
                    
                    # Concatenate role and content for context and output
                    context = f"{current_content}\n{next_role}:{next_content}"
                    
                    memory.save_context({"question": context}, {"output": ""})
            
            general_system_template = r""" 
-You are a helpful assistant. You have been provided with context.
-Answer the question from provided context, do not make up any answer.
-Use the following pieces of data to answer the question at the end.

-Following is the relevant data:
----

{context}

----

Chat History:

------
{chat_history}
------

"""
            general_user_template ="``{question}``"
            messages = [
                        SystemMessagePromptTemplate.from_template(general_system_template), HumanMessagePromptTemplate.from_template(general_user_template)
            ]
            qa_prompt = ChatPromptTemplate.from_messages( messages )


            general_system_template2 = r""" 
Given the following conversation and a follow up question, only rephrase the follow up question a little to be a standalone question, in its original language...

Example1:
Conversation:
User: Please provide the sales data for Q2 2023.
Assistant: Here's the sales data for Q2 2023.

User: And can you also share the profit figures?

Rephrased Standalone Question: 
Human: Please share the profit figures for Q2 2023.

Example 2:
Conversation:
user: What are the top-selling products in the last month?
Assistant: The top-selling products last month were A, B, and C.
User: Could you list the top-performing products in terms of customer reviews as well?



Rephrased Standalone Question:
User: Please list the top-performing products in terms of customer reviews.

Wrong response: 
Sure, here is the rephrased standalone question: Human:

Conversation:

------
{chat_history}
------
Human: {question}

"""
            general_user_template2 ="``{question}``"
            qa_prompt2 = ChatPromptTemplate.from_template( general_system_template2 )

            #llm2 = ChatGooglePalm(temperature=0.8, verbose=True)


                # Load question-answering chain
            chain = ConversationalRetrievalChain.from_llm(  
            llm , verbose= True, memory = memory, 
            retriever=vectorstore.as_retriever(search_kwargs={'k': 4}), max_tokens_limit=1000#,condense_question_prompt= qa_prompt2, condense_question_llm=llm2
            ,combine_docs_chain_kwargs={'prompt':qa_prompt})
            #chain = ConversationalRetrievalChain.from_llm(GooglePalm(temperature=0.5), verbose= True, prompt = PROMPT,memory=memory, chain_type="stuff")
                
            #chain = load_qa_chain(ChatOpenAI(temperature=0.9, model="gpt-3.5-turbo-0613", streaming=True) , verbose= True, prompt = PROMPT, memory=memory,chain_type="stuff")
            

            with st.chat_message("assistant"):
                st_cb = StreamlitCallbackHandler(st.container(), expand_new_thoughts=True)
                response = chain({"question": prompt,   
                                        "chat_history": st.session_state['history']}, callbacks=[st_cb])
                st.session_state.messages.append({"role": "Assistant", "content": response['answer']})
                st.session_state['history'].append((prompt, response["answer"])) 
                st.write(response['answer'])



hide_streamlit_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            </style>
            """
st.markdown(hide_streamlit_style, unsafe_allow_html=True) 
    

