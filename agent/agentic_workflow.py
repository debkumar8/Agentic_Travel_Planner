from utils.model_loader import ModelLoader # inside this file we write the code for loading the model
from prompt_library.prompt import SYSTEM_PROMPT
from langgraph.graph import StateGraph, MessagesState, END, START
from langgraph.prebuilt import ToolNode, tools_condition


from tools.weather_info_tool import WeatherInfoTool
from tools.place_search_tool import PlaceSearchTool
from tools.expense_calculator_tool import CalculatorTool
from tools.currency_conversion_tool import CurrencyConverterTool



class GraphBuilder():
    def __init__(self,model_provider: str = "groq"):
        self.model_loader = ModelLoader(model_provider=model_provider)
        self.llm = self.model_loader.load_llm()
        
        self.tools = []
        
        self.weather_tools = WeatherInfoTool()
        self.place_search_tools = PlaceSearchTool()
        self.calculator_tools = CalculatorTool()
        self.currency_converter_tools = CurrencyConverterTool()
        
        self.tools.extend([* self.weather_tools.weather_tool_list, 
                           * self.place_search_tools.place_search_tool_list,
                           * self.calculator_tools.calculator_tool_list,
                           * self.currency_converter_tools.currency_converter_tool_list]) # complete list of the tools
        
        self.llm_with_tools = self.llm.bind_tools(tools=self.tools) # bind it with the LLM
        
        self.graph = None
        
        self.system_prompt = SYSTEM_PROMPT
    

# we can say it as brain
    def agent_function(self,state: MessagesState):    # This is message type state
        """Main agent function"""   # doc string is the description of the function
        user_question = state["messages"]
        input_question = [self.system_prompt] + user_question  # This is my user question, based "input_question", it will choose appropriate tool
        response = self.llm_with_tools.invoke(input_question) # Bind the tool with the llm, after that invoke it and i am passing my questions
        return {"messages": [response]}
 # "input_question" is a collection of question, as well as system prompt, what ever will generate i will written that as a state
 # This particular thing either going to the tool or it is "END" the process, if it is basic question
 # If it will going to the tool, it will generate the final answer with the tool calling, then after completion it will "END" the final process 

    def build_graph(self):
        graph_builder=StateGraph(MessagesState)
        graph_builder.add_node("agent", self.agent_function)  # I have the agent function like brain
        graph_builder.add_node("tools", ToolNode(tools=self.tools))
        graph_builder.add_edge(START,"agent")   # we are passing input to the agent
        graph_builder.add_conditional_edges("agent",tools_condition) # Agent to tool condition
        graph_builder.add_edge("tools","agent") # tool to agent , means creating loop again
        graph_builder.add_edge("agent",END)     # agent is done with final output, then we will be ending the process 
        self.graph = graph_builder.compile()
        return self.graph
        
    def __call__(self):
       return self.build_graph()

# we create here the entire langgraph work flow

