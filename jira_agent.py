from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional
from contextlib import asynccontextmanager
import os
import re
from fastapi.middleware.cors import CORSMiddleware
import json
from dotenv import load_dotenv
from langchain_community.utilities.jira import JiraAPIWrapper
from langchain_community.agent_toolkits.jira.toolkit import JiraToolkit
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from utils.jira_pipeline import fetch_jira_tokens 
from langchain.agents import create_agent

load_dotenv()

# Pydantic models
class JiraQueryRequest(BaseModel):
    query: str
    use_fallback: bool = True 

class JiraQueryResponse(BaseModel):
    response: str
    query_used: str
    method_used: str 
    success: bool

# Create the lifespan context manager
@asynccontextmanager
async def lifespan(app: FastAPI):
    initialize_jira_components()
    yield

# Define app with lifespan
app = FastAPI(title="Jira Agent API", version="1.0.0", lifespan=lifespan)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables
jira_agent = None
jira_wrapper = None
llm = None
jql_generation_prompt = None
summarization_prompt = None
jira_pipeline = None  # <-- New global for JiraPipeline

def initialize_jira_components():
    """Initialize all Jira and LangChain components"""
    global jira_agent, jira_wrapper, llm, jql_generation_prompt, summarization_prompt, jira_pipeline
    
    print("Initializing Jira and LangChain components...")
    
    # Initialize LangChain Jira components
    jira_wrapper = JiraAPIWrapper()
    toolkit = JiraToolkit.from_jira_api_wrapper(jira_wrapper)
    tools = toolkit.get_tools()
    print(tools)
    
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash-lite",
        temperature=0,
    )

    system_prompt = """You are a specialized Jira assistant.
You MUST use the provided tools to answer questions about Jira.
Do NOT answer any questions from your own knowledge.
If a user's query seems like a general knowledge question, you MUST assume it refers to data within Jira.

*** CRITICAL JQL RULE ***
When filtering by a field with a string value that contains spaces (like a person's name, a project name, or a summary), you MUST enclose the value in single or double quotes.
CORRECT: assignee = 'Aru Sharma'
INCORRECT: assignee = Aru Sharma
CORRECT: summary ~ '"New Login Button"'
INCORRECT: summary ~ 'New Login Button'

Begin!"""

    
    jira_agent = create_agent(
        llm,
        tools=tools,
        system_prompt=system_prompt,
        debug=True,
    )
    
    jql_generation_prompt = PromptTemplate.from_template(
            """You are an expert in Jira Query Language (JQL). Your sole task is to convert a user's natural language request into a valid JQL query.
        You must only respond with the JQL query string and nothing else.

        --- Important JQL Syntax Rules ---
        1.  **Quoting:** Any string value containing spaces or special characters MUST be enclosed in single ('') or double ("") quotes.
            -   Example: `assignee = 'Aru Sharma'`
            -   Example: `summary ~ '"Detailed new feature"'`
        2.  **Usernames:** When searching for an assignee, it is best to use their name in quotes.
        3.  **Linked Issues:** Use `issue in linkedIssues('KEY')`, not `issueLink = KEY`
        4.  **Operators:** Use `=` for single values, `IN` for multiple.
        5.  **Dates:** Example: `created < '2019-01-01'`
        5. **Case:** Fields lowercase, keywords uppercase
        - Example: status = 'Open' ORDER BY created DESC
        6. **Allowed fields:** project, status, assignee, reporter, issuetype,
        priority, created, updated, resolution, labels, summary, description

        --- Examples ---
        User Request: "find all tickets in the 'PROJ' project"
        JQL Query: project = 'PROJ'

        User Request: "show me all open bugs in the 'Mobile' project assigned to Aru Sharma"
        JQL Query: project = 'Mobile' AND issuetype = 'Bug' AND status = 'Open' AND assignee = 'Aru Sharma'

        User Request: "what were the top 5 highest priority issues created last week?"
        JQL Query: created >= -7d ORDER BY priority DESC

        Now, convert the following user request into a JQL query.

        User Request: "{user_query}"
        JQL Query:"""
)
    
    summarization_prompt = PromptTemplate.from_template(
            """You are a helpful assistant. The user asked the following question:

        "{user_query}"

        An AI agent attempted to answer this but failed. As a fallback, we ran a JQL query and got the following raw Jira issue data.
        Please analyze this data and provide a clear, concise, and helpful answer to the user's original question. If the data seems irrelevant or empty, state that you couldn't find relevant information.

        JSON Data:
        {json_data}

        Based on the data, answer the user's question.
        """
    )

    # Initialize JiraPipeline
    # jira_pipeline = JiraPipeline(
    #     server_url=os.getenv("JIRA_INSTANCE_URL"),
    #     username=os.getenv("JIRA_USERNAME"),
    #     token=os.getenv("JIRA_API_TOKEN")
    # )


def validate_and_fix_jql(jql: str) -> str:
    """
    Validates and fixes common JQL mistakes:
    - Ensures string values with spaces are quoted
    - Fixes linked issue syntax
    - Cleans up extra quotes
    - Normalizes ORDER BY
    """
    fixed_jql = jql.strip()

    # Fix linkedIssues syntax
    fixed_jql = re.sub(r"issueLink\s*=\s*([A-Z]+-\d+)", r"issue in linkedIssues('\1')", fixed_jql)

    # Quote assignee with spaces
    fixed_jql = re.sub(
        r"assignee\s*=\s*([A-Za-z]+\s+[A-Za-z]+)",
        lambda m: f"assignee = '{m.group(1)}'",
        fixed_jql
    )

    # Quote project with spaces
    fixed_jql = re.sub(
        r"project\s*=\s*([A-Za-z]+\s+[A-Za-z]+)",
        lambda m: f"project = '{m.group(1)}'",
        fixed_jql
    )

    # Quote summary search with spaces
    fixed_jql = re.sub(
        r"summary\s*~\s*([^'\"]\S+)",
        lambda m: f"summary ~ '\"{m.group(1)}\"'",
        fixed_jql
    )

    # Normalize IN clauses
    fixed_jql = re.sub(r"\(\s*", "(", fixed_jql)
    fixed_jql = re.sub(r"\s*\)", ")", fixed_jql)
    fixed_jql = re.sub(r",\s*", ", ", fixed_jql)

    # Uppercase operators
    keywords = ["order by", "and", "or", "not", "in", "is", "empty"]
    for kw in keywords:
        fixed_jql = re.sub(rf"\b{kw}\b", kw.upper(), fixed_jql, flags=re.IGNORECASE)


    # Remove double quotes errors
    fixed_jql = fixed_jql.replace("''", "'").replace('""', '"')
    fixed_jql = re.sub(r"[;.,]+$", "", fixed_jql)

    # Normalize ORDER BY
    fixed_jql = re.sub(r"order by", "ORDER BY", fixed_jql, flags=re.IGNORECASE)

    return fixed_jql


def intelligent_agent_run(query: str):
    """
    Tries to run the main agent. If it fails, it uses an LLM to generate
    a JQL query from the user's input and executes that instead using JiraPipeline.
    """
    try:
        print("--- Attempting main agent execution ---")
        response = jira_agent.invoke({
            "messages": [
                {"role": "user", "content": query}
            ]
            })
        # return {
        #     "response": response,
        #     "method_used": "agent",
        #     "query_used": query,
        #     "success": True
        # }
        print("result :- ", response["messages"][-1].content)

    except Exception as e:
        print("\n--- Agent failed, switching to intelligent fallback mode ---")
        print(f"Error: {e}\n")

        # Use the LLM to generate a JQL query from the user's original query
        print("Generating JQL from natural language...")
        jql_generation_chain = jql_generation_prompt | llm
        generated_jql = jql_generation_chain.invoke({"user_query": query}).content
        print(f"Dynamically Generated JQL: '{generated_jql}'")

        fixed_jql = validate_and_fix_jql(generated_jql)
        print(f"Validated & Fixed JQL: {fixed_jql}")

        # Execute the generated JQL query using JiraPipeline
        print("Fetching and normalizing Jira data via pipeline...")
        try:
            fallback_data = fetch_jira_tokens(fixed_jql)
            print("fallback_data:", fallback_data)
            # df = jira_pipeline.normalize_data(df)
            # fallback_data = df.to_dict(orient="records")

            if not fallback_data:
                return {
                    "response": "The generated JQL query ran successfully but returned no issues. Please try rephrasing your request or be more specific.",
                    "method_used": "fallback",
                    "query_used": fixed_jql,
                    "success": True
                }

            # Summarize results for the user
            print("Summarizing JiraPipeline results for the user...")
            summarization_chain = summarization_prompt | llm
            final_response = summarization_chain.invoke({
                "user_query": query,
                "json_data": fallback_data 
            }).content

            print("Final summarized response generated.", final_response)
            
            return {
                "response": final_response,
                "method_used": "fallback",
                "query_used": fixed_jql,
                "success": True
            }

        except Exception as fallback_e:
            print(f"Fallback JQL execution also failed: {fallback_e}")
            return {
                "response": f"I'm sorry, I couldn't process your request. Both the primary agent and the fallback query failed. The last error was: {fallback_e}",
                "method_used": "failed",
                "query_used": query,
                "success": False
            }

initialize_jira_components()
print("Jira Agent Ready. Type 'exit' to quit.\n")

while True:
    user_input = input("You: ")
    if user_input.lower() in {"exit", "quit"}:
        break

    result = intelligent_agent_run(user_input)