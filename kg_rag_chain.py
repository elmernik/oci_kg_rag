from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_models import ChatOCIGenAI
from langchain_community.vectorstores import OracleVS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.retrievers.document_compressors import FlashrankRerank
from langchain.retrievers import ContextualCompressionRetriever
from langchain_community.vectorstores.utils import DistanceStrategy
import oracledb
from config_private import USER, PWD, DSN, WLOC, WPWD, COMPARTMENT_OCID
from custom_retriever import KnowledgeGraphRetriever


def create_kg_rag_chain(top_k, k_after_rerank, get_next_node=False, next_node_k=0, parent_depth=0, child_depth=0):
    """Create the rag chain on the knowledge graph table"""

    # Connect to database
    connection = oracledb.connect(user=USER, password=PWD, dsn=DSN,
                                    wallet_location=WLOC, wallet_password=WPWD)

    # Embeddings model
    embed_model= HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-l6-v2")

    # Init vector store
    vector_store = OracleVS(client=connection, embedding_function=embed_model, table_name="KG", 
                            distance_strategy=DistanceStrategy.COSINE)

    # Retriever 
    retriever = KnowledgeGraphRetriever(vector_store=vector_store, 
                    search_kwargs={"k": top_k, "get_next_node": get_next_node, 
                                   "next_node_k": next_node_k, "parent_depth": parent_depth, "child_depth": child_depth})

    # Compressor/Reranker
    compressor = FlashrankRerank(top_n=k_after_rerank)
    compression_retriever = ContextualCompressionRetriever(base_compressor=compressor, base_retriever=retriever)

    # Initialize llm for chat
    chatllm = ChatOCIGenAI(
        model_id="cohere.command-r-plus",
        service_endpoint="https://inference.generativeai.eu-frankfurt-1.oci.oraclecloud.com",
        compartment_id=COMPARTMENT_OCID,
        model_kwargs={"temperature": 0.3, "max_tokens": 500},
    )

    # Contextualize question
    contextualize_q_system_prompt = (
        "Given a chat history and the latest user question "
        "which might reference context in the chat history, "
        "formulate a standalone question which can be understood "
        "without the chat history. Do NOT answer the question, "
        "just reformulate it if needed and otherwise return it as is."
    )

    contextualize_q_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", contextualize_q_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )

    history_aware_retriever = create_history_aware_retriever(
        chatllm, compression_retriever, contextualize_q_prompt
    )

    # Answer question
    system_prompt = (
        "You are an assistant for question-answering tasks. "
        "Use the following pieces of retrieved context "
        "in the form of knowledge graph triplets answer to "
        "the question. If you don't know the answer, say that you "
        "don't know. Use three sentences maximum and keep the "
        "answer concise."
        "\n\n"
        "{context}"
    )

    qa_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )

    question_answer_chain = create_stuff_documents_chain(chatllm, qa_prompt)

    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)


    # Statefully manage chat history
    store = {}


    def get_session_history(session_id: str) -> BaseChatMessageHistory:
        if session_id not in store:
            store[session_id] = ChatMessageHistory()
        return store[session_id]


    conversational_rag_chain = RunnableWithMessageHistory(
        rag_chain,
        get_session_history,
        input_messages_key="input",
        history_messages_key="chat_history",
        output_messages_key="answer",
    )

    return conversational_rag_chain


