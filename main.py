from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from agent.agentic_workflow import GraphBuilder
# from utils.save_to_document import save_document
from starlette.responses import JSONResponse
import os
import datetime
from dotenv import load_dotenv
from pydantic import BaseModel
load_dotenv()


app = FastAPI() # created as object as FastAPI

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # set specific origins in prod
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
class QueryRequest(BaseModel):
    question: str

@app.post("/query") # post something throughout the form
async def query_travel_agent(query:QueryRequest):
    try:
        print(query)
        graph = GraphBuilder(model_provider="groq")  # "GraphBuilder" object is created
        react_app=graph() 
        # we call "GraphBuilder" as a function, if i call this object as function, 
        # automatically it calls "__call__" from agent.agentic_workflow import GraphBuilder and going to be written "build_graph ()"
        #react_app = graph.build_graph()

        png_graph = react_app.get_graph().draw_mermaid_png() # the graph is going to be saved
        with open("my_graph.png", "wb") as f:
            f.write(png_graph)

        print(f"Graph saved as 'my_graph.png' in {os.getcwd()}")
        # Assuming request is a pydantic object like: {"question": "your text"}
        messages={"messages": [query.question]} # getting query from state
        output = react_app.invoke(messages) # invoking it

        # If result is dict with messages:
        if isinstance(output, dict) and "messages" in output:
            final_output = output["messages"][-1].content  # Last AI response
        else:
            final_output = str(output)
        
        return {"answer": final_output} # what ever final output we will get, we written that
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})
    

# This is very simple end point 