import streamlit as st
from src.langgraphagenticai.ui.streamlitui.loadui import LoadStreamlitUI
from src.langgraphagenticai.LLMS.groqllm import GroqLLM
from src.langgraphagenticai.graph.graph_builder import GraphBuilder
from src.langgraphagenticai.ui.streamlitui.display_result import DisplayResultStreamlit

def load_langgraph_agenticai_app():
    st.set_page_config(page_title="LangGraph Agentic AI", layout="wide")

    # Load UI
    ui = LoadStreamlitUI()
    user_input = ui.load_streamlit_ui()

    if not user_input:
        st.error("Error: Failed to load user input from the UI.")
        return

    usecase = user_input.get("selected_usecase")

    if not usecase:
        st.error("Error: No use case selected.")
        return

    # Get LLM
    try:
        obj_llm_config = GroqLLM(user_contols_input=user_input)
        model = obj_llm_config.get_llm_model()
        if not model:
            st.error("Error: LLM could not be initialized.")
            return
    except Exception as e:
        st.error(f"Error: LLM config failed - {e}")
        return

    # Setup graph
    try:
        graph_builder = GraphBuilder(model)
        graph = graph_builder.setup_graph(usecase)
    except Exception as e:
        st.error(f"Error: Graph setup failed - {e}")
        return

    # Run selected use case
    if usecase == "AI News" and st.session_state.get("IsFetchButtonClicked"):
        frequency = st.session_state.get("timeframe")
        DisplayResultStreamlit(usecase, graph, frequency).display_result_on_ui()

    elif usecase in ["Basic Chatbot", "Chatbot With Web"]:
        user_message = st.chat_input("Enter your message:")
        if user_message:
            DisplayResultStreamlit(usecase, graph, user_message).display_result_on_ui()
