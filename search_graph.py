import oracledb
from langchain.embeddings import HuggingFaceEmbeddings
import array
from show_graph import show_graph
from config_private import user, pwd, dsn, wloc, wpwd


def search_graph(query, get_related_to_object1, get_related_to_object2):
    # Embed query
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-l6-v2")
    embed_query = array.array("f", embeddings.embed_query(query))

    # Define SQL
    vec_search_sql = '''SELECT OBJECT_1, RELATION, OBJECT_2 
        FROM KG 
        ORDER BY VECTOR_DISTANCE(vec, :query_vector, COSINE) 
        FETCH EXACT FIRST 5 ROWS ONLY'''

    object1_sql = '''SELECT OBJECT_1, RELATION, OBJECT_2 
        FROM KG WHERE OBJECT_1 = :query_object'''

    object2_sql = '''SELECT OBJECT_1, RELATION, OBJECT_2 
        FROM KG WHERE OBJECT_2 = :query_object'''

    # Connect to database
    connection = oracledb.connect(user=user, password=pwd, dsn=dsn,
                                wallet_location=wloc, wallet_password=wpwd)

    # Get cursor
    cursor = connection.cursor()

    # Print 5 closest matches by cosine distance
    result = []
    for row in cursor.execute(vec_search_sql, [embed_query]):
        print(row)
        result.append(list(row))

    # Get everything else immediately related to each object 1 !!! TODO This logic should be improved !!!
    if get_related_to_object1:
        object1s = set([i[0] for i in result])
        for o1 in object1s:
            for row in cursor.execute(object1_sql, [o1]):
                result.append(list(row))

    # Get everything else immediately related to each object 2 !!! TODO This logic should be improved !!!
    if get_related_to_object2:
        object2s = set([i[2] for i in result])
        for o2 in object2s:
            for row in cursor.execute(object2_sql, [o2]):
                result.append(list(row))

    connection.close()
    return result


if __name__ == "__main__":
    # Define query
    question = "Who is Warren Buffets father?"

    # Define if you want to get other nodes
    gro1 = False
    gro2 = False

    # Get result
    result = search_graph(question, gro1, gro2)

    # Show a partial knowledge graph
    head, relation, tail = list(zip(*result))
    show_graph(head, relation, tail)