from llama_index import SimpleDirectoryReader, download_loader, LLMPredictor, GPTSimpleKeywordTableIndex, PromptHelper, ServiceContext
from langchain import OpenAI

SimpleDirectoryReader = download_loader("SimpleDirectoryReader")
loader = SimpleDirectoryReader("SGG\DocsGPT\docs", recursive=True, exclude_hidden=True)
docs = loader.load_data()

# Create a predictor
llm_predictor = LLMPredictor(llm=OpenAI(temperature=0.9, model_name="text-davinci-003"))

max_input_size = 4096
num_output = 256
max_chunk_overlap = 20
promthelper = PromptHelper(max_input_size, num_output, max_chunk_overlap)

servicecontext = ServiceContext.from_defaults(llm_predictor=llm_predictor, promthelper=promthelper)

# Create an index

index = GPTSimpleKeywordTableIndex.from_documents(docs, servicecontext=servicecontext)