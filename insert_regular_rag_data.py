from langchain_community.document_loaders.oracleai import OracleTextSplitter, OracleDocLoader
import oracledb
from langchain.embeddings import HuggingFaceEmbeddings
import array
from config_private import user, pwd, dsn, wloc, wpwd

# Read txt file
with open("./example_data/oracle_wikipedia.txt", "r") as f:
    doc_content = "".join(f.readlines())

# Connect to database
connection = oracledb.connect(user=user, password=pwd, dsn=dsn,
                                wallet_location=wloc, wallet_password=wpwd)

# Get cursor
cursor = connection.cursor()

# Insert document data 
cursor.execute("INSERT INTO DOCUMENTS VALUES (1, 'oracle_wikipedia.txt', :data)", [doc_content])

# loading from Oracle Database table (Slightly unnecessary to do it like this but I wanted to test it out)
loader_params = {
    "owner": "GRAPH_RAG",
    "tablename": "DOCUMENTS",
    "colname": "DATA",
}

# Load the docs
loader = OracleDocLoader(conn=connection, params=loader_params)
docs = loader.load()

# split by default parameters
splitter_params = {"normalize": "all"}

# get the splitter instance
splitter = OracleTextSplitter(conn=connection, params=splitter_params)

# Get the chunks
list_chunks = []
for doc in docs:
    chunks = splitter.split_text(doc.page_content)
    list_chunks.extend(chunks)

# Initialize embedding model and embed chunks
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-l6-v2")
list_vectors = [array.array("f", embeddings.embed_query(" ".join(chunk))) for chunk in list_chunks]

# Generate IDs
ids = [i for i in range(1, len(list_chunks)+1)]

# Insert data
data = list(zip(*[ids, list_chunks, list_vectors]))
cursor.executemany("INSERT INTO DOC_CHUNKS VALUES (:1, 'oracle_wikipedia.txt', :2, :3)", data)

# Commit and close connection
connection.commit()
connection.close()

