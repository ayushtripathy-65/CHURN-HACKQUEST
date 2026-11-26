"""
Root entrypoint for Streamlit Cloud deployment.
Forwards execution to HackQuest/app.py.
"""
import os
import sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
HACKQUEST_DIR = os.path.join(BASE_DIR, "HackQuest")
if HACKQUEST_DIR not in sys.path:
    sys.path.insert(0, HACKQUEST_DIR)

# Change working directory so all relative paths resolve
os.chdir(HACKQUEST_DIR)

# Run HackQuest/app.py
app_path = os.path.join(HACKQUEST_DIR, "app.py")
with open(app_path, "r", encoding="utf-8") as f:
    code = f.read()
exec(compile(code, app_path, "exec"))
