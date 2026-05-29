import os
import sys
import subprocess

if __name__ == "__main__":
    script_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "src", "gui", "dashboard.py")
    subprocess.call([sys.executable, "-m", "streamlit", "run", script_path])
