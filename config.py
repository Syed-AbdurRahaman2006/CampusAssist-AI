"""Application configuration for CampusAssist AI."""

import os

APP_TITLE = "CampusAssist AI"
APP_DESCRIPTION = "AI-powered conversational campus helpdesk prototype"
APP_VERSION = "0.1.0"

# Path to the FAQ dataset (same directory as this file)
DATA_PATH = os.path.join(os.path.dirname(__file__), "data.json")

# Minimum similarity score (0–1) to consider a FAQ match valid
MATCH_THRESHOLD = 0.45
