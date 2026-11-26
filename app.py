"""
Root entrypoint for Streamlit Community Cloud.
Ensures seamless deployment whether Streamlit Cloud is pointed to 'app.py' or 'HackQuest/app.py'.
"""
import os
import sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
HACKQUEST_DIR = os.path.join(BASE_DIR, "HackQuest")

if HACKQUEST_DIR not in sys.path:
    sys.path.insert(0, HACKQUEST_DIR)

# Switch working directory to HackQuest so relative paths (data/..., artifacts/...) resolve cleanly
os.chdir(HACKQUEST_DIR)

# Execute HackQuest app
app_path = os.path.join(HACKQUEST_DIR, "app.py")
with open(app_path, "r", encoding="utf-8") as f:
    code = compile(f.read(), app_path, "exec")
exec(code, globals())
