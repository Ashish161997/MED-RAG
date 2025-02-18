from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from langchain.llms import HuggingFacePipeline
from langchain.chains import RetrievalQA
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from transformers import BitsAndBytesConfig
import torch

from IPython.display import display, Markdown
def colorize_text(text):
    for word, color in zip(["Reasoning", "Question", "Answer", "Total time"], ["blue", "red", "green", "magenta"]):
        text = text.replace(f"{word}:", f"\n\n**<font color='{color}'>{word}:</font>**")
    return text

# 1. Load the model with quantization
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16
)

model_name = "microsoft/phi-1_5"
tokenizer = AutoTokenizer.from_pretrained(model_name)

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",
    trust_remote_code=True,
    quantization_config=bnb_config
)

# 2. Wrap the model using Hugging Face pipeline
pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=100,  # Adjust based on needs
    do_sample=True  # Enables non-deterministic sampling
)

llm = HuggingFacePipeline(pipeline=pipe)

# 3. Load embeddings
try:
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
except Exception as ex:
    print("Exception: ", ex)

# 4. Load and split documents
loader = PyPDFLoader(r'A:\rag\RAG_medical\medical_oncology_handbook_june_2020_edition.pdf')
documents = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
texts = text_splitter.split_documents(documents)

# 5. Create Chroma vector DB
vectordb = Chroma.from_documents(documents=texts, embedding=embeddings, persist_directory="chroma_db")

print("Vector Database is Created")

# 6. Create retriever
retriever = vectordb.as_retriever()

# 7. Initialize RetrievalQA with the correct model
qa = RetrievalQA.from_chain_type(
    llm=llm,  # Pass the HuggingFacePipeline wrapped model
    chain_type="stuff",
    retriever=retriever,
    verbose=True
)



def test_rag(qa, query):
    """
    Test the Retrieval Augmented Generation (RAG) system.
    
    Args:
        qa (RetrievalQA.from_chain_type): Langchain function to perform RAG
        query (str): query for the RAG system
    Returns:
        None
    """

    
    response = qa.run(query)

    full_response =  f"Question: {query}\nAnswer: {response}"
    display(Markdown(colorize_text(full_response)))


query = "what is medical oncology?"
test_rag(qa, query)