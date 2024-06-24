import oracledb
from langchain.embeddings import HuggingFaceEmbeddings
import array
from show_graph import show_graph
from config_private import user, pwd, dsn, wloc, wpwd


def vector_search_graph(query, top_n=5):
    """Function for searching the graph using vector similarity"""
    # Embed query
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-l6-v2")
    embed_query = array.array("f", embeddings.embed_query(query))

    # Define SQL
    vec_search_sql = f'''SELECT OBJECT_1, RELATION, OBJECT_2 
        FROM KG 
        ORDER BY VECTOR_DISTANCE(vec, :query_vector, COSINE) 
        FETCH EXACT FIRST {top_n} ROWS ONLY'''

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

    connection.close()
    return result


def get_next_nodes(start_node, depth=1, parent_or_child="child"):
    """Function for getting the next nodes (either parent or child)
    from a specific starting node until a certain depth.
    """
    # Default to child
    poc = 1 if parent_or_child == "child" else (2 if parent_or_child == "parent" else 1)

    # Define sql query depending on whether you want parent or child nodes
    get_nodes_sql = f'''SELECT OBJECT_1, RELATION, OBJECT_2 
        FROM KG WHERE OBJECT_{poc} = :query_object'''

    # Connect to database
    connection = oracledb.connect(user=user, password=pwd, dsn=dsn,
                                wallet_location=wloc, wallet_password=wpwd)
    cursor = connection.cursor()

    def _fetch_nodes(cursor, current_node, depth, current_depth, sql, poc):
        """Recursive helper function for fetching the nodes"""
        # If depth has been reached then stop
        if current_depth > depth:
            return []
        
        # Get next nodes
        cursor.execute(sql, [current_node])
        results = cursor.fetchall()

        all_results = results[:]

        # Recursion on the next node
        for next_parent, _, next_child in results:
            if poc == 1:
                all_results.extend(_fetch_nodes(cursor, next_child, depth, current_depth + 1, sql, poc))
            else:
                all_results.extend(_fetch_nodes(cursor, next_parent, depth, current_depth + 1, sql, poc))
        return all_results
    
    # Call helper function and return result from recursion
    try:
        next_nodes = _fetch_nodes(cursor, start_node, depth, 1, get_nodes_sql, poc)
    finally:
        connection.close()
    
    return next_nodes


if __name__ == "__main__":
    # Define query
    question = "Who is Warren Buffets father?"

    # Get result
    #result = search_graph(question, gro1, gro2)
    result = get_next_nodes("businessman", 1, "parent")

    # Show a partial knowledge graph
    head, relation, tail = list(zip(*result))
    show_graph(head, relation, tail)