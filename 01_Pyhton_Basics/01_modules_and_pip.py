"""
Day 1: Modules, Pip & Environment Setup
---------------------------------------
Demonstrating:
1. Comments (Single-line & Multi-line docstrings)
2. Built-in standard library modules (os, time, random)
3. External package via pip (requests)
4. System sound execution
"""


# 1. BUILT-IN MODULES (System & Utilities)

import os
import time
import random

# Single-line comment: Inspecting system environment

print("=" * 45)
print("1. BUILT-IN MODULES DEMO")
print("=" * 45)
print(f"Current Directory: {os.getcwd()}")
lucky_number = random.randint(1, 100)
print(f"Randomly generated number: {lucky_number}")


# 2. EXTERNAL MODULE VIA PIP (requests)

print("\n" + "=" * 45)
print("2. EXTERNAL MODULE (PIP) DEMO")
print("=" * 45)

try:
    import requests
    response = requests.get("https://api.github.com", timeout=5)
    print(f"GitHub API Status Code: {response.status_code} (Success)")
except ImportError:
    print("Run `pip install requests` to test external packages.")


# 3. AUDIO VERIFICATION (macOS Native Voice)

import subprocess
import time

# 1. Hardware audio pipe ko pehle se wake up karo (silent cue)

subprocess.run(["say", "[[volm 0.0]] wake [[volm 1.0]]"])
time.sleep(0.3)

# 2. Main speech

speech_text = "Hello Raj, modules and pip setup is complete."
print(f"Speaking: '{speech_text}'")
subprocess.run(["say", "-r", "150", speech_text])

print("Execution finished successfully!")