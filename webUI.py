import os

import streamlit as st
from llama_index import download_loader
from llama_index.node_parser import SimpleNodeParser
from llama_index import GPTVectorStoreIndex
from llama_index import LLMPredictor, GPTVectorStoreIndex, PromptHelper, ServiceContext
from langchain import OpenAI
from presidio_anonymizer import AnonymizerEngine
from presidio_anonymizer.entities import RecognizerResult, OperatorConfig, OperatorResult


os.environ["openai_secret_key"] == st.secrets["openai_secret_key"]

doc_path = './data/'
index_file = 'index.json'

if 'response' not in st.session_state:
    st.session_state.response = ''

def send_click():
    index = GPTVectorStoreIndex.from_documents(documents)
    query_engine = index.as_query_engine()
    st.session_state.response  = query_engine.query(st.session_state.prompt)

index = None

st.image('./logo-black.png')
st.title("SmartGPT")

sidebar_placeholder = st.sidebar.container()
uploaded_file = st.file_uploader("Choose a file")

def anonymizer():
    Engine = AnonymizerEngine()
    result = Engine.anonymize(
        text=st.session_state.response,
        analyzer_results=[
            RecognizerResult(
                entity_type="PERSON",
                start=11,
                end=16,
                score=0.8,
            ),
            RecognizerResult(
                entity_type="PHONE_NUMBER",
                start=40,
                end=52,
                score=0.8,
            ),
        ],
        operators={"DEFAULT": OperatorConfig("replace", {"new_value": "TEST"})},
    )
    st.session_state.response = result.text

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
    sidebar_placeholder.write(documents[0].get_text()[:10000]+'...')

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
    sidebar_placeholder.write(documents[0].get_text()[:10000]+'...')

if index != None:
    st.text_input("Ask something: ", key='prompt')
    col1, col2, col3 = st.columns([0.5, 0.5, 0.5])
    with col1:
        st.button("Send", on_click=send_click)
    with col2:
        st.button("Clear", on_click=lambda: st.session_state.clear())
    with col3:
        st.button("Anonymize", on_click=lambda: anonymizer())
    if st.session_state.response:
        st.subheader("Response: ")
        st.success(st.session_state.response, icon= "🤖")

# python -m streamlit run .\webUI.py