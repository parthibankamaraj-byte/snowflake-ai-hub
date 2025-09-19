import os
import sys

# Ensure we can import the package from ./src
CURRENT_DIR = os.path.dirname(__file__)
SRC_DIR = os.path.join(CURRENT_DIR, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from resume_chat_bot_agent.app import app  # FastAPI instance


