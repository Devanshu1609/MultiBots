# # utils/slack_tools.py

import os
from slack_sdk import WebClient
from dotenv import load_dotenv

load_dotenv()

slack_token = os.getenv("SLACK_TOKEN")
slack_client = WebClient(token=slack_token)

def get_channel_info(channel_name: str) -> str:
    """Get info about a Slack channel by name. Input should be a channel name like #general"""
    try:
        response = slack_client.conversations_list()
        channels = response["channels"]
        for channel in channels:
            if channel["name"] == channel_name.lstrip("#"):
                return str(channel)
        return f"Channel '{channel_name}' not found."
    except Exception as e:
        return f"Error: {e}"
    
