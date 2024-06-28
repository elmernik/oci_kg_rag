from typing import List

from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_core.vectorstores import VectorStore
from langchain_core.pydantic_v1 import Field
from langchain_community.vectorstores.utils import DistanceStrategy
import array
import json


class KnowledgeGraphRetriever(BaseRetriever):
    """Class for implementing some additional functionality to the retriever, 
    mainly traversing the next nodes in the knowledge graph to retrieve
    potentially more relevant context
    
    Will hopefully make this a lot less scuffed and much faster in the future"""

    vector_store: VectorStore
    """Has to be a OracleVS"""
    search_kwargs: dict = Field(default_factory=dict)
    """Possible kwargs: "k": int, "get_next_node": bool, 
    "next_node_k": int, "parent_depth": int, "child_depth": int"""
    distance_strategy2function = {
        DistanceStrategy.EUCLIDEAN_DISTANCE: "EUCLIDEAN",
        DistanceStrategy.DOT_PRODUCT: "DOT",
        DistanceStrategy.COSINE: "COSINE",
    }
    """Makes it able to get different distance strategies"""
    
    def _bind_list(self, length: int)-> str:
        """Create a bind list to allow for lists in sql"""
        return ", ".join([f":{i}" for i in range(1, length+1)])


    def _get_next_nodes(
        self, ids: List[int], parent_depth: int = 0, child_depth: int = 0
    )->List[int]:
        """Get the next nodes up to a certain depth either for parents, children or both"""
        def _fetch_nodes(cursor, current_node: str, depth: int, current_depth: int, poc: int):
            """Recursive helper function for fetching the nodes"""
            # If depth has been reached then stop
            if current_depth > depth:
                return []
            
            next_node_query = f"""SELECT ID, OBJECT_1, RELATION, OBJECT_2 
            FROM KG WHERE OBJECT_{poc} = :query_object"""
            
            # Get next nodes
            cursor.execute(next_node_query, [current_node])
            results = cursor.fetchall()

            all_results = results[:]

            # Recursion on the next node
            for _, next_parent, _, next_child in results:
                if poc == 1:
                    all_results.extend(_fetch_nodes(cursor, next_child, depth, current_depth + 1, poc))
                else:
                    all_results.extend(_fetch_nodes(cursor, next_parent, depth, current_depth + 1, poc))
            return all_results
        
        # Query for getting all Object 1 for each already retrieved row
        query = f"""SELECT object_1 
        FROM {self.vector_store.table_name}
        WHERE id IN ({self._bind_list(len(ids))})"""

        with self.vector_store.client.cursor() as cursor:
            # Get the objects
            cursor.execute(query, ids)
            objects = [o[0] for o in cursor.fetchall()]
            nodes = []
            # For each object extend the node list
            for object in objects:
                nodes.extend(_fetch_nodes(cursor, object, parent_depth, 0, 2))
                nodes.extend(_fetch_nodes(cursor, object, child_depth, 0, 2))

        # Return a unique list of ids of the next nodes
        return list(set([node[0] for node in nodes if node[0] not in ids]))


    def _search_graph(
        self, embedding: List[float], k: int = 4, next_node_k: int = 4, 
        get_next_node: bool = False, parent_depth: int = 0, child_depth: int = 0
    )->List[Document]:
        """Searches the graph using vector search and possibly getting further nodes"""
        # Initialize retrieved docs list
        docs = []

        # Vector search query
        base_query = f"""
        SELECT id,
          text,
          metadata,
          vector_distance(embedding, :embedding,
          {self.distance_strategy2function[self.vector_store.distance_strategy]}) as distance
        FROM {self.vector_store.table_name}
        ORDER BY distance
        FETCH APPROX FIRST {k} ROWS ONLY
        """

        # Convert embedding to array dtype
        embedding_arr = array.array("f", embedding)

        with self.vector_store.client.cursor() as cursor:
            # Execute the vector search
            cursor.execute(base_query, [embedding_arr])
            results = cursor.fetchall()

            # If get_next_node then retrieve further connected nodes
            if get_next_node == True:
                # Get ids
                ids = [result[0] for result in results]

                # Retrieved ids for the further connected nodes
                next_nodes_ids = self._get_next_nodes(ids, parent_depth, child_depth)

                # If there were noedes then extend result docs
                if next_nodes_ids != None:
                    next_node_query = f"""
                    SELECT id,
                    text,
                    metadata,
                    vector_distance(embedding, :embedding,
                    {self.distance_strategy2function[self.vector_store.distance_strategy]}) as distance
                    FROM {self.vector_store.table_name}
                    WHERE id IN ({self._bind_list(len(next_nodes_ids))})
                    ORDER BY distance
                    FETCH APPROX FIRST {next_node_k} ROWS ONLY
                    """

                    next_nodes_ids.insert(0, embedding_arr)

                    cursor.execute(next_node_query, next_nodes_ids)
                    results.extend(cursor.fetchall())

            # Get metadata and doc content
            for result in results:
                metadata = json.loads(
                    result[2] if result[2] is not None else "{}"
                )
                doc = Document(
                    page_content=(
                        result[1]
                        if result[1] is not None
                        else ""
                    ),
                    metadata=metadata,
                )
                docs.append(doc)
        # Return list of documents
        return docs

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> List[Document]:
        """Base function for retrieval"""
        # Embed query
        embedding = self.vector_store.embedding_function.embed_query(query)

        # Call search graph and return docs
        docs = self._search_graph(embedding, **self.search_kwargs)
        return docs