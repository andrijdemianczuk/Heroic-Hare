#Python libs
import functools
import os
from typing import Any, Generator, Literal, Optional

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
from pydantic import BaseModel

#You can find the ID in the URL of the genie room /genie/rooms/<GENIE_SPACE_ID>
#In lieu of the locally scoped variable for the PAT, we'll use the one from our secrets store for security.

GENIE_SPACE_ID = "01f026a703761605b18fa1d904cf1a64"
genie_agent_description = "This genie agent can answer any questions around billing and Databricks or AWS related expenses associate with the account."

genie_agent = GenieAgent(
    genie_space_id=GENIE_SPACE_ID,
    genie_agent_name="Genie_DBX_Cost",
    description=genie_agent_description,
    client=WorkspaceClient(
        host=os.getenv("DB_MODEL_SERVING_HOST_URL"),
        token=os.getenv("DATABRICKS_GENIE_PAT")
        #token=secret,
    ),
)

#Multi-agent Genie works best with claude 3.7 or gpt 4o models. Both of these are served using the system.ai.* databricks-uc namespace.
LLM_ENDPOINT_NAME = "databricks-claude-3-7-sonnet"
llm = ChatDatabricks(endpoint=LLM_ENDPOINT_NAME)

############################################################
# You can also create agents with access to additional tools
############################################################
toolbox = []

#If you want to add more tools to to the toolbox, add additional tools and update the description of this agent
uc_tool_names = ["system.ai.*"]
uc_toolkit = UCFunctionToolkit(function_names=uc_tool_names)

toolbox.extend(uc_toolkit.tools)

code_agent_description = (
    "The Coder agent specializes in solving programming challenges, generating code snippets, debugging issues, and explaining complex coding concepts.",
)
code_agent = create_react_agent(llm, tools=toolbox)

#NOTE: This cell is just the DEFINITION of the agent graph. All we're doing here is describing how to compose the composite application.

#Update the max number of iterations between supervisor and worker nodes before returning to the user. This is how many internal 'conversations' the supervisor has with the other agents. This is a maximum value - if an answer is sufficient with fewer iterations, then great.
MAX_ITERATIONS = 5

#Add the description for each agent we're going to use as a dictionary
worker_descriptions = {
    "Genie": genie_agent_description,
    "Coder": code_agent_description,
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
def agent_node(state, agent, name):
    result = agent.invoke(state)
    return {
        "messages": [
            {
                "role": "assistant",
                "content": result["messages"][-1].content,
                "name": name,
            }
        ]
    }


#This is the callable object that contains the response payload.
def final_answer(state):
    prompt = "Using only the content in the messages, respond to the previous user question using the answer given by the other assistant messages."
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
code_node = functools.partial(agent_node, agent=code_agent, name="Coder")
genie_node = functools.partial(agent_node, agent=genie_agent, name="Genie")

#Build the graph from the nodes, including something to send a result back to whatever's invoking the application (aka final answer).
workflow = StateGraph(AgentState)
workflow.add_node("Genie", genie_node)
workflow.add_node("Coder", code_node)
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
