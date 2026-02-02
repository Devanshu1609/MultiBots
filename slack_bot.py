import os
from agno.agent import Agent
from agno.tools.slack import SlackTools
from dotenv import load_dotenv
from agno.models.google import Gemini
from slack_sdk import WebClient
from utils.slack_tools import get_channel_info
load_dotenv()

client = WebClient(token=os.getenv("SLACK_TOKEN"))

Slack_tools = SlackTools()

agent = Agent(
    model=Gemini(id="gemini-2.5-flash-lite"),
    markdown=True,
    tools=[Slack_tools, get_channel_info],
)

channel_id = "C0F5D59L3"

client.conversations_join(channel=channel_id)

agent.print_response(
    "give me detailed information about channel #general",
    markdown=True
)