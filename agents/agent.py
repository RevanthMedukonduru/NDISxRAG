import logging
import operator
from typing import Any, List, Literal, Optional, TypedDict, Annotated
from langchain.schema import Document
from langchain_anthropic import ChatAnthropic
from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent
from pydantic import BaseModel, Field
import pandas as pd
from prompts.prompt_helper import get_custom_prompt_template
from prompts.prompt_definitions import QUERY_DECOMPOSITION_BASED_ON_CHAT_HISTORY_AND_AGENT_CONTEXT_PROMPT, TOOL_SELECTION_PROMPT, SYSTEM_PROMPT_FOR_REPL_TOOL, QUERY_DECOMPOSITION_BASED_ON_HYDE, SEMANTIC_SEARCH_RESULTS_ANALYSIS_PROMPT, SPAM_DETECTION_PROMPT, AGENT_CONTEXT, GENERAL_QUERY_PROCESSING_PROMPT, FINAL_RESPONSE_GENERATOR
from Embedding.vector_store import QdrantStore
from observability.langfuse_tracer import get_langfuse_configuration
from NDISxRAG.settings import LOGGER_NAME

# Logger
logger = logging.getLogger(LOGGER_NAME)

# QdrantStore
vector_store = QdrantStore()

# LLMs
chat_model_gpt = ChatOpenAI(model="gpt-4o-mini")
chat_model_claude = ChatAnthropic(model="claude-3-5-sonnet-20240620")

# Prompts
query_rewrite_prompt = get_custom_prompt_template(
    standalone_prompt_template = QUERY_DECOMPOSITION_BASED_ON_CHAT_HISTORY_AND_AGENT_CONTEXT_PROMPT
)
tool_selection_prompt = get_custom_prompt_template(
    standalone_prompt_template = TOOL_SELECTION_PROMPT
)
hyde_query_decomposition_prompt = get_custom_prompt_template(
    standalone_prompt_template = QUERY_DECOMPOSITION_BASED_ON_HYDE
)
semantic_search_results_analysis_prompt = get_custom_prompt_template(
    standalone_prompt_template = SEMANTIC_SEARCH_RESULTS_ANALYSIS_PROMPT
)
spam_detection_prompt = get_custom_prompt_template(
    standalone_prompt_template = SPAM_DETECTION_PROMPT
)
general_query_processing_prompt = get_custom_prompt_template(
    standalone_prompt_template = GENERAL_QUERY_PROCESSING_PROMPT
)
final_response_generator_prompt = get_custom_prompt_template(
    standalone_prompt_template = FINAL_RESPONSE_GENERATOR
)

# Pydantic Classes
class QueryRewrite(BaseModel):
    """
    Query Rewrite Pydantic Model.
    """
    rewritten_query: str = Field(description="Rewritten query based on the context and chat history.")

class SelectedTool(BaseModel):
    """
    Selected Tool Pydantic Model.
    """
    tool_name: str = Field(description="Name of the selected tool.")

class HydeQueryDecomposition(BaseModel):
    """
    Query Decomposition Pydantic Model.
    """
    subqueries: List[str] = Field(description="List of passages.")

class SemanticSearchResultsAnalysis(BaseModel):
    """
    Semantic Search Results Analysis Pydantic Model.
    """
    analysed_response: str = Field(description="Analysis of the semantic search results.")

class QueryClassification(BaseModel):
    """
    Classification based on the provided query.
    """
    category: Literal["SPAM_PROMPT_INJECTION", "GENERAL", "CONTEXTUAL"]
    reason: Optional[str] = None
    
class GeneralQueryProcessing(BaseModel):
    """
    General Query Processing Pydantic Model.
    """
    response: str = Field(description="Response to the general query.")

class FinalResponse(BaseModel):
    """
    Final Response Pydantic Model.
    """
    response: str = Field(description="Final response to the user query.")
    
# Binded LLMs
structured_output_for_query_rewrite = chat_model_claude.with_structured_output(schema=QueryRewrite)
structured_output_for_tool_selection = chat_model_gpt.with_structured_output(schema=SelectedTool)
structured_output_for_hyde_query_decomposition = chat_model_claude.with_structured_output(schema=HydeQueryDecomposition)
structured_output_for_semantic_search_results_analysis = chat_model_gpt.with_structured_output(schema=SemanticSearchResultsAnalysis)
structured_output_for_spam_detection = chat_model_gpt.with_structured_output(schema=QueryClassification)
structured_output_for_general_query_processing = chat_model_gpt.with_structured_output(schema=GeneralQueryProcessing)
structured_output_for_final_response = chat_model_claude.with_structured_output(schema=FinalResponse)

# Chains
query_rewrite_chain = query_rewrite_prompt | structured_output_for_query_rewrite
tool_selection_chain = tool_selection_prompt | structured_output_for_tool_selection
hyde_query_decomposition_chain = hyde_query_decomposition_prompt | structured_output_for_hyde_query_decomposition
semantic_search_results_analysis_chain = semantic_search_results_analysis_prompt | structured_output_for_semantic_search_results_analysis
spam_detection_chain = spam_detection_prompt | structured_output_for_spam_detection
general_query_processing_chain = general_query_processing_prompt | structured_output_for_general_query_processing
final_response_generator_chain = final_response_generator_prompt | structured_output_for_final_response

# Agents
def get_repl_agent(df: pd.DataFrame) -> Any:
    """
    Returns an agent that can analyze the given DataFrame.
    """
    repl_tool_analysis_agent = create_pandas_dataframe_agent(
        llm=chat_model_gpt,
        df=df,
        max_iterations=5,
        verbose=True,
        allow_dangerous_code=True,
        return_intermediate_steps=True,
        agent_type="tool-calling",
        prefix=SYSTEM_PROMPT_FOR_REPL_TOOL,
        include_df_in_prompt=True,
    )
    return repl_tool_analysis_agent

# Agent State
class AgentState(TypedDict):
    """
    Agent State.
    """
    original_query: str
    query_category: str
    agent_context: str
    rewritten_independent_query: str
    subqueries: List[str]
    dataframe: pd.DataFrame
    chat_history: List[str]
    intermediate_steps: List[tuple[str, Any]]
    final_response: str
    citations: List[str]

# ----- Helper Functions -----
def _format_chat_history(chat_history: List[str]) -> str:
    """
    Format the chat history for display purposes.
    
    chat_history: List of chat messages.
    ex: [{'role': 'user', 'content': 'hello'}, {'role': 'assistant', 'content': 'hi'} ...]
    """
    return "\n".join([f"{msg['role']}: {msg['content']}" for msg in chat_history])

def _get_formatted_dataframe(dataframe: pd.DataFrame, documents: List[Document]) -> tuple[str, str]:
   """
   Get the formatted DataFrame for display purposes.
   """
   # Group by source/filename and sheet
   file_sheet_rows = {}
   for doc in documents:
       source = doc.metadata["source"]
       sheet = doc.metadata["sheet"]
       row_no = doc.metadata["row_no"]
       
       if source not in file_sheet_rows:
           file_sheet_rows[source] = {}
           
       if sheet not in file_sheet_rows[source]:
           file_sheet_rows[source][sheet] = []
           
       file_sheet_rows[source][sheet].append(str(row_no))

   # Collect all rows
   all_rows = [dataframe.iloc[doc.metadata["row_no"]] for doc in documents]
   
   # Format citations
   citation_parts = []
   for source, sheets in file_sheet_rows.items():
       for sheet, rows in sheets.items():
           citation_parts.append(f"*Filename:*{source}: *Sheet name:* {sheet}: *Row No's:* [{', '.join(rows)}]")
   
   citations = "\n".join(citation_parts)
   
   return str(pd.DataFrame(all_rows).to_markdown()), citations

# ----- Nodes -----
def general_query_or_spam_filter(data: AgentState) -> AgentState:
    """
    This function filters out general queries or spam messages (Like a guard node).
    """
    # Query categorization
    query_category = spam_detection_chain.invoke(input={"original_query": data["rewritten_independent_query"], "agent_context": data["agent_context"]})
    
    # Update the query category
    data["query_category"] = query_category.category
    
    # Update the intermediate steps
    data["intermediate_steps"].append(("Query Classification", query_category.category))
    return data
    
def query_rewrite_node(data: AgentState) -> AgentState:
    """
    Rewrites the input query based on the context and the chat history.
    """
    # Extract the input query and chat history
    original_query = data["original_query"]
    chat_history = _format_chat_history(data["chat_history"])
    agent_context = data["agent_context"]
    
    # Decompose the query based on the chat history
    rewritten_independent_query = query_rewrite_chain.invoke(input={
        "original_query": original_query,
        "agent_context": agent_context,
        "chat_history": chat_history
    })
    
    # Update the intermediate steps
    data["intermediate_steps"].append(("Query Rewrite", rewritten_independent_query.rewritten_query))
    
    # Update the rewritten query
    data["rewritten_independent_query"] = rewritten_independent_query.rewritten_query  
    return data

def query_decomposition_node(data: AgentState) -> AgentState:
    """
    Decomposes the query into subqueries if query is complex or it involves multiple steps.
    
    This function is a placeholder for the actual implementation.
    """
    data["subqueries"] = [data["rewritten_independent_query"]]
    return data

def tool_selection_node(data: AgentState) -> AgentState:
    """
    Selects the most suitable tool based on the rewritten query.
    """
    # Extract the rewritten query
    rewritten_query = data["rewritten_independent_query"]
    
    # Select the most suitable tool based on the rewritten query
    selected_tool = tool_selection_chain.invoke(input={
        "query": rewritten_query,
        "data_format": data["dataframe"].head().to_markdown(),
        "agent_context": data["agent_context"]
    })
    
    # Update the intermediate steps
    data["intermediate_steps"].append(("Tool Selection", selected_tool.tool_name))    
    return data
    
def tool_execution_node(data: AgentState) -> AgentState:
    """
    Executes the selected tool on the provided data.
    """
    # Extract the selected tool
    selected_tool = data["intermediate_steps"][-1][1]
    result = []
    
    # Execute the selected tool on the provided data
    if selected_tool == "python_repl_ast":
        # Get the REPL tool agent
        repl_tool_agent = get_repl_agent(data["dataframe"])
        
        # Execute the REPL tool agent
        repl_agent_outcome = repl_tool_agent.invoke(data["rewritten_independent_query"], handle_parsing_errors=True)
        
        # update the intermediate steps
        result = [("Tool Execution/Intermediate steps/Repl", repl_agent_outcome["intermediate_steps"]),("Tool Execution/Repl", repl_agent_outcome["output"])]
        
        # get the sheet name from the dataframe
        data["citations"] = f"Sheet name: {data['dataframe'].sheet_name}"
    elif selected_tool == "semantic_search_tool_for_dataframe":
        # Apply Hyde to decompose query - to get more semantic match
        decomposed_passages = hyde_query_decomposition_chain.invoke(input={
            "original_query": data["rewritten_independent_query"],
            "agent_context": data["agent_context"],
            "no_of_passages": 4
        })
        
        # Retrieve the results/semantic search results
        is_connected = vector_store.reconnect_or_open_connection()
        logger.info("[Inside Tool Execution Node] Connection to Qdrant: %s", is_connected)
        semantic_search_results = vector_store.retrieve_data(collection_name="data", queries=decomposed_passages.subqueries, n_results=4, top_k=10)
        is_closed = vector_store.close_connection()
        logger.info("[Inside Tool Execution Node] Connection to Qdrant closed: %s", is_closed)
        
        # Get the Dataframe based on Semantic Search Results
        formatted_dataframe, formatted_citations = _get_formatted_dataframe(data["dataframe"], semantic_search_results)
        
        # Analyse the results of the semantic search tool
        semantic_search_results_analysis = semantic_search_results_analysis_chain.invoke(input={
            "query": data["original_query"],
            "semantic_search_results": formatted_dataframe
        })
        
        # Update the intermediate steps
        result = [("Tool Execution/Intermediate steps/Semantic Search/", formatted_dataframe), ("Tool Execution/Semantic Search", semantic_search_results_analysis.analysed_response)] 
        
        # update the citations
        data["citations"] = formatted_citations
               
    # Update the intermediate steps
    data["intermediate_steps"].extend(result)    
    return data

def final_response(data: AgentState) -> AgentState:
    """
    Analyzes the results of the tool execution and provides the final response.
    """
    # Retrieve the draft response
    logger.info("Final Response: %s", data)
    last_intermediate_step = data["intermediate_steps"][-1]
    draft_response = last_intermediate_step[1]
    
    # Generate the final response
    if data["query_category"] == "CONTEXTUAL":
        data["final_response"] = final_response_generator_chain.invoke(input={"original_query": data["rewritten_independent_query"], "draft_response": draft_response, "agent_context": data["agent_context"]}).response
    else:
        data["final_response"] = draft_response
    return data      

def spam_query_processor(data: AgentState) -> AgentState:
    """
    Processes the spam query.
    """
    # Update the intermediate steps
    data["intermediate_steps"] = data["intermediate_steps"] + [("Spam Query Processing", "Sorry, I cannot answer this query.")]
    return data

def general_query_processor(data: AgentState) -> AgentState:
    """
    Processes the general query.
    """
    # Process the general query
    general_query_response = general_query_processing_chain.invoke(input={"original_query": data["original_query"], "agent_context": data["agent_context"]})
    
    # Update the intermediate steps
    data["intermediate_steps"].append(("General Query Processing", general_query_response.response))
    return data
  
def query_router(data: AgentState) -> AgentState:
    """
    Routes the query based on the query category.
    """
    if data["query_category"] == "SPAM_PROMPT_INJECTION":
        return "Spam_Query_Processor"
    elif data["query_category"] == "GENERAL":
        return "General_Query_Processor"
    else:
        return "Query_Decomposition"
    
# ----- Agent -----
def graph_builder():
    """
    Builds the state graph for the agent.
    """
    # Create the state graph
    workflow = StateGraph(AgentState)
    
    # Add the nodes to the state graph
    workflow.add_node("Query_Classification", general_query_or_spam_filter)
    workflow.add_node("Query_Router", query_router)
    workflow.add_node("Spam_Query_Processor", spam_query_processor)
    workflow.add_node("General_Query_Processor", general_query_processor)
    workflow.add_node("Query_Rewrite", query_rewrite_node)
    workflow.add_node("Query_Decomposition", query_decomposition_node)
    workflow.add_node("Tool_Selection", tool_selection_node)
    workflow.add_node("Tool_Execution", tool_execution_node)
    workflow.add_node("Final_Response", final_response)
    
    # Add the edges to the state graph
    workflow.set_entry_point("Query_Rewrite")
    
    # Route based on query category
    workflow.add_conditional_edges(
        "Query_Classification",
        query_router,
        {
            "Spam_Query_Processor": "Spam_Query_Processor",
            "General_Query_Processor": "General_Query_Processor",
            "Query_Decomposition": "Query_Decomposition"
        }
    )
    
    # Flow 1: Contextual Query
    workflow.add_edge("Query_Rewrite", "Query_Classification")
    workflow.add_edge("Query_Decomposition", "Tool_Selection")
    workflow.add_edge("Tool_Selection", "Tool_Execution")
    workflow.add_edge("Tool_Execution", "Final_Response")
    
    # Flow 2: Spam Query
    workflow.add_edge("Spam_Query_Processor", "Final_Response")
    
    # Flow 3: General Query
    workflow.add_edge("General_Query_Processor", "Final_Response")
    
    # Finish the workflow
    workflow.add_edge("Final_Response", END)
    
    # Compile the state graph
    app = workflow.compile()
    return app

# Run the agent
def generate_response(user_id: int, file_path: str, original_query: str, chat_history: List[str]) -> dict:
    """
    Run the agent.
    """
    logger.info("Generating response for user: %s", user_id)
    
    # Start Qdrant connection
    vector_store.reconnect_or_open_connection()
    
    # Load the excel file
    dataframe = pd.read_excel(file_path)
    
    # Create the state graph
    app = graph_builder()
    
    # Initialize the agent state
    agent_state = {
        "original_query": original_query,
        "query_category": "",
        "rewritten_independent_query": "",
        "subqueries": [],
        "dataframe": dataframe,
        "chat_history": chat_history,
        "intermediate_steps": [],
        "final_response": "",
        "agent_context": AGENT_CONTEXT
    }
    
    # Get Langfuse Configuration
    langfuse_configuration = get_langfuse_configuration(user_id=user_id, run_title="NDISxRAG")
    
    # Run the agent
    logger.info("Running the agent... %s", agent_state)
    final_state = app.invoke(agent_state, config=langfuse_configuration)
    logger.info("Agent execution completed... %s", final_state)
    return final_state["final_response"], final_state["citations"]