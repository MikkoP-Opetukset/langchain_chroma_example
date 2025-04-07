# Documentation for LangChain
# https://python.langchain.com/docs/introduction/


#%%
import os
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

#%%
# Create a loader that loads .txt files from `data/` directory
# By changing the loader_cls you can change what type of files you are loading
loader = DirectoryLoader(path="data/", glob="*.txt", loader_cls=TextLoader,
                         show_progress=True)

#%%
# Use the loader to actually load the files

# This might produce a warning "libmagic is unavailable but assists in filetype
# detection. Please consider installing libmacic for better results." This
# should work even without libmagic, but if you want to install it you have to
# google how to do it on your OS.
docs = loader.load()

#%%
# Showing example of what the first 100 characters in the first document are
print(docs[0].page_content[:100])

#%%
# Splitting the documents
# First we define how we want to split the texts into smaller chunks.
# Here we use character length as the length_function, size tries to be close
# to 150, the text overlaps by 20 characters, and we specify our list of
# characters that we use as acceptable splitting point.

# The is_separator_regex and separators are not strictly necessary. The default
# is ["\n\n", "\n", " ", ""] so we add a positive lookbehind for the pattern
# r"(?<=\. )" to make the splitter favor the end of sentence instead of end of
# words. This is because in the source material of this example the text is
# just one line without any new lines. This pattern is better than using a
# simple ". " separator because that would not include the characters ". " in
# the split, and the dot and space have meaning in text. This is to
# demonstrate, that in some cases you need to add your own logic to the default
# values, depending on your source data.
text_splitter = RecursiveCharacterTextSplitter(
    length_function=len,
    chunk_size=150,
    chunk_overlap=20,
    is_separator_regex=True,
    separators=["\n\n", "\n", r"(?<=\. )", " ", ""]
)

splits = text_splitter.split_documents(docs)

#%%
# Embedding
# Instead of HuggingFaceEmbeddings we could use any other Embedding model
# loader, like OpenAIEmbeddings or OllamaEmbeddings.

# In encode_kwargs you can set your preferred parameters for encoding.
# When using HuggingFaceEmbeddings refer to documentation on what can be set in
# here:
# https://sbert.net/docs/package_reference/sentence_transformer/SentenceTransformer.html#sentence_transformers.SentenceTransformer.encode
encode_kwargs = {"normalize_embeddings": True}

# In model_kwargs you can set your preferred settings for the embeddings.
# When using HuggingFaceEmbeddings refer to documentation on what can be set in
# here:
# https://sbert.net/docs/package_reference/sentence_transformer/SentenceTransformer.html#sentence_transformers.SentenceTransformer

# If you have a cuda capable gpu you can try uncommenting this next line and
# the commented line in the embeddings.
# model_kwargs = {"device": "cuda"}

# The model I chose for this is the BAAI/bge-m3 model from HuggingFace
embeddings = HuggingFaceEmbeddings(
    model_name="BAAI/bge-m3",
    encode_kwargs=encode_kwargs,
#    model_kwargs=model_kwargs,
    show_progress=True
)
#%%
# Creating the vector store
# Here we use Chroma, but this could be any vector store of your choosing. For
# large projects Chroma might not be suitable, Milvus for example might be a
# better choice when dealing with very large amounts of data.

if not os.path.exists("./data/chroma_db"):
    os.makedirs("./data/chroma_db")

vector_store = Chroma(
    collection_name="my_collection",
    embedding_function=embeddings,
    persist_directory="./data/chroma_db"
)
vector_store.add_documents(splits)
print("")

#%%
# Example of similarity search
results = vector_store.similarity_search(
    query="When was Kate Winslet born?",
    k=4
)

for res in results:
    print(f"{res.page_content} | {res.metadata}")
#%%
results = vector_store.similarity_search(
    query="Who is Woody Allen?",
    k=4
)

for res in results:
    print(f"{res.page_content} | {res.metadata}")

#%%
results = vector_store.similarity_search(
    query="What is Pascal?",
    k=4
)

for res in results:
    print(f"{res.page_content} | {res.metadata}")

#%%
