import oracledb
from langchain.embeddings import HuggingFaceEmbeddings
import array
from config_private import user, pwd, dsn, wloc, wpwd

# Connect to database
connection = oracledb.connect(user=user, password=pwd, dsn=dsn,
                                wallet_location=wloc, wallet_password=wpwd)

# Get cursor
cursor = connection.cursor()

# File to read kg data from
file_path = "./example_data/oracle_wikipedia_kg.txt"

# Define the heads, relations, and tails
with open(file_path, "r") as f:
   # TODO Maybe make this a little less scuffed
   lines = [[item.strip("'") for item in line.strip("[]\n").split("', '")] for line in f.readlines()]
   unzipped = list(zip(*lines))
   head, relation, tail = unzipped[0], unzipped[1], [i.strip("'],") for i in unzipped[2]]

# Embed each object1-relation-object2 into a vector
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-l6-v2")
vectors = [array.array("f", embeddings.embed_query(" ".join(line))) for line in lines]

# Generate IDs
ids = [i for i in range(1, len(lines)+1)]

# Insert data
data = list(zip(*[ids, list(head), list(relation), list(tail), vectors]))
cursor.executemany("INSERT INTO KG VALUES (:1, :2, :3, :4, :5)", data)

# Commit and close connection
connection.commit()
connection.close()
