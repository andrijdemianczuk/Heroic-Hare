#Python libs
import functools
import os
from typing import Any, Generator, Literal, Optional, Type
import requests
import feedparser

#Databricks sdk & Databricks langchain implementation
from databricks.sdk import WorkspaceClient
from databricks_langchain import (
    ChatDatabricks,
    UCFunctionToolkit,
)
from databricks_langchain.genie import GenieAgent

#Langchain tools (langraph is our agent lib)
from langchain_core.runnables import RunnableLambda
from langgraph.graph import END, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.prebuilt import create_react_agent
from langchain.agents import AgentType, initialize_agent, agent
from langchain.tools import Tool
from langchain.tools import BaseTool
# from langchain.chat_models import ChatOpenAI
from langchain.chains.conversation.memory import ConversationBufferWindowMemory

#MLflow stuff
import mlflow
from mlflow.langchain.chat_agent_langgraph import ChatAgentState
from mlflow.pyfunc import ChatAgent
from mlflow.types.agent import (
    ChatAgentChunk,
    ChatAgentMessage,
    ChatAgentResponse,
    ChatContext,
)

#Parsing libs
from pydantic import BaseModel, Field

#This is the foundation LLM that we'll be using for the basis of our agents
LLM_ENDPOINT_NAME = "databricks-claude-3-7-sonnet"
llm = ChatDatabricks(endpoint=LLM_ENDPOINT_NAME, extra_params={"temperature": 0.3})

#Input Schema for the weather class. We'll be using this to compose the tool
class WeatherInput(BaseModel):
    latitude: float = Field(..., description="Latitude of the location")
    longitude: float = Field(..., description="Longitude of the location")

#Create an implementation inheriting the BaseTool abstract class from LangChain. _run() and _arun() are both required implementations. The name and description attributes are also required.
class FetchWeatherTool(BaseTool):
    name: str = "fetch_weather"
    description: str = "Fetch hourly weather temperature data for a given latitude and longitude. When asked about weather, assume that the user is always referring to temperature only. temperature_2m refers to the temperature in degrees celsius."
    args_schema: Type[BaseModel] = WeatherInput #This is the input schema we defined above.

    #The _run() function is always called when invoked. It's essentially doing the same thing as a class constructor, but since we're invoking the class from a toolchain in lieu of instancing the class as an object, this is run instead. This behaviour is what we're inheriting from the BaseTool() class.
    def _run(self, latitude: float, longitude: float):
        url = "https://api.open-meteo.com/v1/forecast"
        params = {
            "latitude": latitude,
            "longitude": longitude,
            "hourly": "temperature_2m",
        }
        response = requests.get(url, params=params)
        if response.status_code == 200:
            return response.json()["hourly"]["temperature_2m"][:5]  # Preview
        return f"Failed to fetch data: {response.status_code}"

    def _arun(self, *args, **kwargs):
        raise NotImplementedError("Async not supported.")

class ArxivSearchInput(BaseModel):
    query: str = Field(..., description="Search keywords for the arXiv research papers")
    max_results: Optional[int] = Field(5, description="Max number of papers to return")

class SearchArxivTool(BaseTool):
    name: str = "search_arxiv"
    description: str = "Search for academic papers on arXiv related to a given keyword."
    args_schema: Type[BaseModel] = ArxivSearchInput

    def _run(self, query: str, max_results: int = 3):
        url = "http://export.arxiv.org/api/query"
        params = {
            "search_query": f"all:{query}",
            "start": 0,
            "max_results": max_results,
        }
        response = requests.get(url, params=params)
        feed = feedparser.parse(response.text)
        results = []
        for entry in feed.entries[:max_results]:
            results.append(f"{entry.title} — {entry.link}")
        return "\n".join(results) if results else "No papers found."

    def _arun(self, *args, **kwargs):
        raise NotImplementedError("Async not supported.")

class AssetSearchInput(BaseModel):
    asset_query: str = Field(..., description="The name of the location or asset to search for")

class SearchAssetsTool(BaseTool):
    name: str = "search_assets"
    description: str = "Search for geographic or structural asset metadata using OpenStreetMap. "
    args_schema: Type[BaseModel] = AssetSearchInput

    def _run(self, asset_query: str):
        url = "https://nominatim.openstreetmap.org/search"
        params = {"q": asset_query, "format": "json"}
        headers = {"User-Agent": "LangChainAgent/1.0 (andrij.demianczuk@databricks.com)"}

        response = requests.get(url, params=params, headers=headers)
        if response.status_code != 200:
            return f"Search API returned {response.status_code}. Cannot continue using search_assets."
        
        data = response.json()
        if data:
            result = data[0]
            return f"{result['display_name']} (lat: {result['lat']}, lon: {result['lon']})"
        else:
            return f"No results found for '{asset_query}'"

    def _arun(self, *args, **kwargs):
        raise NotImplementedError("Async not supported.")

weather_tools = [
    FetchWeatherTool(),
    SearchArxivTool(),
]

weatherdoc_agent_description = (
    "The Weather and Document agent specializes in retrieving weather (temperature) information from known coordinates. This agent can also retrieve documents from arxiv with a keyword search. This agent focuses on returning useful information to the prompter for weather and research articles. It can have internal conversations to get the most complete infomration from it's tools.",
)

weather_doc_conversational_memory = ConversationBufferWindowMemory(
    memory_key='chat_history',
    k=5,
    return_messages=True
)

weatherdoc_bot = initialize_agent(
    tools=weather_tools,
    llm=llm,
    agent=AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True,
    max_iterations=10
)

asset_tools = [
    SearchAssetsTool()
]

asset_agent_description = (
    "The asset agent looks up details for asset searches. Typically this will return coordinates but also contains information about the timezone for the assets as well. This input epxects a string for the query and is generally adept at major landmarks in North America.",
)

asset_conversational_memory = ConversationBufferWindowMemory(
    memory_key='chat_history',
    k=10,
    return_messages=True
)

asset_bot = initialize_agent(
    tools=asset_tools,
    llm=llm,
    agent=AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True,
    max_iterations=10
)

#Update the max number of iterations between supervisor and worker nodes before returning to the user. This is how many internal 'conversations' the supervisor has with the other agents. This is a maximum value - if an answer is sufficient with fewer iterations, then great.
MAX_ITERATIONS = 10

#Add the description for each agent we're going to use as a dictionary
worker_descriptions = {
    "Weatherdoc_Agent":weatherdoc_agent_description,
    "Asset_Agent":asset_agent_description,
}

#Flatten the descriptions into a single string variable
formatted_descriptions = "\n".join(
    f"- {name}: {desc}" for name, desc in worker_descriptions.items()
)

#Tell the LM in plain language about the agents it has access to
system_prompt = f"Decide between routing between the following workers or ending the conversation if an answer is provided. \n{formatted_descriptions}"
options = ["FINISH"] + list(worker_descriptions.keys())
FINISH = {"next_node": "FINISH"}

#Make use of all the above definitions and create the supervisor. This is what we'll be interfacing with and logging in MLFlow.
def supervisor_agent(state):
    count = state.get("iteration_count", 0) + 1
    if count > MAX_ITERATIONS:
        return FINISH
    
    #Define our chaining logic
    class nextNode(BaseModel):
        next_node: Literal[tuple(options)]

    #Assemble the entire chain, defining the supervisor and callable agents with some simple recursion logic.
    preprocessor = RunnableLambda(
        lambda state: [{"role": "system", "content": system_prompt}] + state["messages"]
    )
    supervisor_chain = preprocessor | llm.with_structured_output(nextNode)
    next_node = supervisor_chain.invoke(state).next_node
    
    #If the response routed back to the same node, exit the loop. This identifies when the conversation has reached its peak epoch.
    if state.get("next_node") == next_node:
        return FINISH
    return {
        "iteration_count": count,
        "next_node": next_node
    }

#This is the function that composes the message that interfaces with the LLM.
def agent_node(state, agent, name, tools):
    prompt = agent.agent.create_prompt(tools=tools)
    agent.agent.llm_chain.prompt = prompt
    user_messages = [msg for msg in state["messages"] if msg["role"] == "user"]
    if not user_messages:
        raise ValueError("No user message found to use as input")

    last_user_message = user_messages[-1]["content"]
    result = agent.invoke({"input": last_user_message})

    return {
        "messages": [
            {
                "role": "assistant",
                "content": f"The {name} has determined: {result['output']}",
                "name": name,
            }
        ]
    }


#This is the callable object that contains the response payload.
def final_answer(state):
    prompt = f"""You are the final summarizer.

                Here is a conversation between a user and multiple assistant agents.

                Each assistant agent was responsible for solving part of the user's question.

                Your job is to answer the original question as clearly as possible based only on the assistant messages below.

                Do not say "I don't know" unless no assistant provided a relevant answer.
                If multiple assistants provided conflicting or redundant answers, pick the most confident and relevant one.

                Respond to the user's original question with the best possible answer.
                """

    preprocessor = RunnableLambda(
        lambda state: state["messages"] + [{"role": "user", "content": prompt}]
    )
    final_answer_chain = preprocessor | llm
    return {"messages": [final_answer_chain.invoke(state)]}


#This object definition is technically just a struct to keep tabs on the agent.
class AgentState(ChatAgentState):
    next_node: str
    iteration_count: int

#Use a functools wrapper to build out the actual agent objects based on their descriptors
weatherdoc_node = functools.partial(agent_node, agent=weatherdoc_bot, name="Weatherdoc_Agent", tools=weather_tools)
assetdoc_node = functools.partial(agent_node, agent=asset_bot, name="Asset_Agent", tools=asset_tools)

#Build the graph from the nodes, including something to send a result back to whatever's invoking the application (aka final answer).
workflow = StateGraph(AgentState)
workflow.add_node("Weatherdoc_Agent", weatherdoc_node)
workflow.add_node("Asset_Agent", assetdoc_node)
workflow.add_node("supervisor", supervisor_agent)
workflow.add_node("final_answer", final_answer)

workflow.set_entry_point("supervisor")
# We want our workers to ALWAYS "report back" to the supervisor when done
for worker in worker_descriptions.keys():
    workflow.add_edge(worker, "supervisor")

# Let the supervisor decide which next node to go
workflow.add_conditional_edges(
    "supervisor",
    lambda x: x["next_node"],
    {**{k: k for k in worker_descriptions.keys()}, "FINISH": "final_answer"},
)
workflow.add_edge("final_answer", END)
multi_agent = workflow.compile()

class LangGraphChatAgent(ChatAgent):
    #Class constructor. This defines how the LangGraphChatAgent is initialized.
    def __init__(self, agent: CompiledStateGraph):
        self.agent = agent

    #This function is a behaviour that returns a response. It defines the chat structure between the agents. I.E., how they talk back and forth with the supervisor agent. We should probably create an installable library for this since it's pretty typical and can benefit from override and extension functionality.
    def predict(
        self,
        messages: list[ChatAgentMessage],
        context: Optional[ChatContext] = None,
        custom_inputs: Optional[dict[str, Any]] = None,
    ) -> ChatAgentResponse:
        request = {
            "messages": [m.model_dump_compat(exclude_none=True) for m in messages]
        }

        messages = []
        for event in self.agent.stream(request, stream_mode="updates"):
            for node_data in event.values():
                messages.extend(
                    ChatAgentMessage(**msg) for msg in node_data.get("messages", [])
                )
        return ChatAgentResponse(messages=messages)

    #This behaviour is how the supervisor keeps track of internal conversations. This is important as it allows agents to pass context to one another.
    def predict_stream(
        self,
        messages: list[ChatAgentMessage],
        context: Optional[ChatContext] = None,
        custom_inputs: Optional[dict[str, Any]] = None,
    ) -> Generator[ChatAgentChunk, None, None]:
        request = {
            "messages": [m.model_dump_compat(exclude_none=True) for m in messages]
        }
        for event in self.agent.stream(request, stream_mode="updates"):
            for node_data in event.values():
                yield from (
                    ChatAgentChunk(**{"delta": msg})
                    for msg in node_data.get("messages", [])
                )

#Create the agent object, and specify it as the agent object to use when loading the agent back for inference via mlflow.models.set_model()
mlflow.langchain.autolog()
AGENT = LangGraphChatAgent(multi_agent)
mlflow.models.set_model(AGENT)
