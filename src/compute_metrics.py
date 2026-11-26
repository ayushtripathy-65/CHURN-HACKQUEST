"""
Root entrypoint for compute_metrics.py
Redirects to HackQuest/src/compute_metrics.py
"""
import os
import sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
HACKQUEST_SRC = os.path.join(os.path.dirname(BASE_DIR), "HackQuest", "src")
if os.path.exists(HACKQUEST_SRC):
    sys.path.insert(0, os.path.join(os.path.dirname(BASE_DIR), "HackQuest"))

from src.compute_metrics import run_compute_metrics

if __name__ == "__main__":
    run_compute_metrics()
