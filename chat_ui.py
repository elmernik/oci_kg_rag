import streamlit as st
from kg_rag_chain import create_kg_rag_chain
from langchain_core.messages import HumanMessage, AIMessage


def reset_chat(init=False):
    """Initialize session variables"""
    st.session_state.chat_history = []
    if init:
        st.session_state.session_id = 1
    else:
        st.session_state.session_id += 1


@st.cache_resource
def init_rag_chain():
    """Create the actual rag chain"""
    return create_kg_rag_chain()


def format_and_show_context(context):
    """Format and show retrieved context to the sidebar"""
    for item in context:
        st.sidebar.markdown(item.page_content)
        st.sidebar.markdown(item.metadata)


def update_chat_history():
    """Update the UI with the chat"""
    for msg in st.session_state.chat_history:
        if isinstance(msg, HumanMessage):
            role = "user"
        else:
            role = "assistant"
        
        with st.chat_message(role):
            st.markdown(msg.content)


st.title("Knowledge Graph RAG")

# Clear chat and get new session if button pressed
if st.sidebar.button("Clear Chat"):
    reset_chat()

# Init history and session
if "chat_history" not in st.session_state:
    reset_chat(init=True)

with st.spinner("Initializing RAG chain..."):
    # Initialize the chain
    rag_chain = init_rag_chain()

# User prompt
if question := st.chat_input("Your message"):
    update_chat_history()

    st.chat_message("user").markdown(question)

    with st.spinner("Calling RAG Chain..."):
        # Call the chain
        response = rag_chain.invoke(
            {"input": question},
            config={
                "configurable": {"session_id": str(st.session_state.session_id)}
            },  # constructs a key "session_id" in `store`.
            )
        
        # Show answer
        answer = response["answer"] 
        st.chat_message("assistant").markdown(answer)

        # Show context
        format_and_show_context(response["context"])

        # Update history
        st.session_state.chat_history.append(HumanMessage(content=question))
        st.session_state.chat_history.append(AIMessage(content=answer))