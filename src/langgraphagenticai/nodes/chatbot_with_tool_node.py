from src.langgraphagenticai.state.state import State


class ChatbotWithToolNode:
    """
    Chatbot logic enhanced with tool integration.
    """

    def __init__(self, model):
        self.llm = model

    def process(self, state: State) -> dict:
        """
        Processes the input state and generates a response with tool integration.
        """
        user_input = state["messages"][-1] if state["messages"] else ""
        llm_response = self.llm.invoke([{"role": "user", "content": user_input}])

        # Simulate tool-specific logic
        tools_response = f"Tool integration for: '{user_input}'"

        return {"messages": [llm_response, tools_response]}


    def create_chatbot(self,tools):
        """
        return a chatbot node function
        """
        llm_with_tools = self.llm.bind_tools(tools)

        def chatbot_node(state:State):
            """
            chatbot logic for the processing the input state and returning the response
            """
            return {"messages":[llm_with_tools.invoke(state["messages"])]}
        
        return chatbot_node
    

    def save_result(self, state):
        frequency = self.state['frequency']
        summary = self.state['summary']
        filename = f"./AINews/{frequency}_summary.md"
        with open(filename, 'w') as f:
            f.write(f"# {frequency.capitalize()} AI News Summary\n\n")
            f.write(summary)
            self.state['filename'] = filename
            return self.state
