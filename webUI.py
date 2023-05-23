import os
import openai
import streamlit as st
import re
from llama_index import download_loader
from llama_index.node_parser import SimpleNodeParser
from llama_index import GPTVectorStoreIndex
from llama_index import LLMPredictor, GPTVectorStoreIndex, PromptHelper, ServiceContext
from langchain import OpenAI
from presidio_anonymizer import AnonymizerEngine
from presidio_anonymizer.entities import RecognizerResult, OperatorConfig, OperatorResult

openai.api_key = st.secrets["OPENAI_API_KEY"]

doc_path = './data/'
index_file = 'index.json'

if 'response' not in st.session_state:
    st.session_state.response = ''

# prompt=f"""You are an ethical specialist and your role is to tell if the documents that are given to you are ethical or not. When you give a response you have to start your sentence with: As an ethical specialist, ..."""
prompt=f"""Tu es un spécialiste éthique et ton rôle est d'analyser la demande faite et d'y répondre clairement en donnant l'article ou le passage de la charte qui correspond a cette demande et en disant très clairement si la demande est réalisable ou non."""

def send_click():
    index = GPTVectorStoreIndex.from_documents(documents)
    query_engine = index.as_query_engine()
    st.session_state.response  = query_engine.query(prompt + st.session_state.prompt)

def check_ethical():
    prompt2 = f"""You are an ethical specialist and your role is to tell if the documents that are given to you are ethical or not by just saying Yes, this chart is ethical or No, this chart is not ethical.
    """
    index = GPTVectorStoreIndex.from_documents(documents)
    query_engine = index.as_query_engine()
    st.session_state.response2  = query_engine.query(prompt2)
    st.success(st.session_state.response2, icon="🚨")
    
def multiple_replacer(*key_values):
    replace_dict = dict(key_values)
    replacement_function = lambda match: replace_dict[match.group(0)]
    pattern = re.compile("|".join([re.escape(k) for k, v in key_values]), re.M)
    return lambda string: pattern.sub(replacement_function, string)

def multiple_replace(string, *key_values):
    return multiple_replacer(*key_values)(string)
# def check_privacy():
#     prompt3 = f"""You are a privacy specialist and your role is to tell if the documents that are given to you are respecting the privacy of the people and the enterprise or not by just saying Yes, this chart is respecting the privacy of the people or No, this chart is not respecting the privacy of the people.
#     """
#     index = GPTVectorStoreIndex.from_documents(documents)
#     query_engine = index.as_query_engine()
#     st.session_state.response3  = query_engine.query(prompt3)
#     st.success(st.session_state.response3, icon="✅")
    

index = None

st.image('./logo-black.png')
st.title("SmartGPT")

sidebar_placeholder = st.sidebar.container()
uploaded_file = st.file_uploader("Drag and drop your ethical chart here")
SimpleDirectoryReader = download_loader("SimpleDirectoryReader")
loader = SimpleDirectoryReader(doc_path, recursive=True, exclude_hidden=True)
documents = loader.load_data()
def replace_entities():
    replaced_text = documents[0].get_text().replace('Renault', '<ENTREPRISE>')

    st.success(replaced_text)

if uploaded_file is not None:

    doc_files = os.listdir(doc_path)
    for doc_file in doc_files:
        os.remove(doc_path + doc_file)

    bytes_data = uploaded_file.read()
    with open(f"{doc_path}{uploaded_file.name}", 'wb') as f: 
        f.write(bytes_data)

    SimpleDirectoryReader = download_loader("SimpleDirectoryReader")

    loader = SimpleDirectoryReader(doc_path, recursive=True, exclude_hidden=True)
    documents = loader.load_data()
    sidebar_placeholder.header('Current Processing Document:')
    sidebar_placeholder.subheader(uploaded_file.name)
    #sidebar_placeholder.write(documents[0].get_text()[:10000]+'...')
    st.success("Chart uploaded succesfully !", icon="✅")
    #check_ethical()
    # check_privacy()
    
    llm_predictor = LLMPredictor(llm=OpenAI(temperature=0, model_name="text-davinci-003"))

    max_input_size = 4096
    num_output = 256
    max_chunk_overlap = 20
    prompt_helper = PromptHelper(max_input_size, num_output, max_chunk_overlap)

    service_context = ServiceContext.from_defaults(llm_predictor=llm_predictor, prompt_helper=prompt_helper)

    index = GPTVectorStoreIndex.from_documents(
        documents, service_context=service_context
    )

elif os.path.exists(index_file):
    index = GPTVectorStoreIndex.load_from_disk(index_file)

    SimpleDirectoryReader = download_loader("SimpleDirectoryReader")
    loader = SimpleDirectoryReader(doc_path, recursive=True, exclude_hidden=True)
    documents = loader.load_data()
    doc_filename = os.listdir(doc_path)[0]
    sidebar_placeholder.header('Current Processing Document:')
    sidebar_placeholder.subheader(doc_filename)
    #sidebar_placeholder.write(documents[0].get_text()+'...')

def set_text():
    st.session_state.response = documents[0].get_text()

if index != None:
    uploaded_file1 = st.file_uploader("Drop a list of entities to anonymize")

    if uploaded_file1 is not None:
        st.success("Entities uploaded succesfully !", icon="✅")
        bytes_data = uploaded_file1.read()
        with open(f"{doc_path}{uploaded_file1.name}", 'wb') as f:
            f.write(bytes_data)
        with open(f"{doc_path}{uploaded_file1.name}", 'r') as f:
            entities = f.read().splitlines()
        a = documents[0].get_text()[:1000]
        b = {entities[i]: '<ENTREPRISE>' for i in range(len(entities))}
        for x,y in b.items():
            a = a.replace(x,y)
        sidebar_placeholder.write(a)
    
    st.text_input("Ask something: ", key='prompt')
    col1, col2 = st.columns([0.5, 0.5])
    with col1:
        st.button("Send", on_click=send_click)
    with col2:
        st.button("Clear", on_click=lambda: st.session_state.clear())
    if st.session_state.response:
        st.subheader("Response: ")
        st.success(st.session_state.response, icon= "🤖")

# python -m streamlit run .\webUI.py