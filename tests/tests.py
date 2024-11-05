import sys
import os
import time
from colorama import init, Fore
init(autoreset=True)

# Add the src directory to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from text_clustering import ClusterClassifier

failures = 0

cc = ClusterClassifier()

def test_func(func):
    global failures
    try:
        start = time.time()
        func()
        print(Fore.GREEN + f"{func} succeeded. Time taken: {time.time() - start:.2f}s")
    except Exception as e:
        print(Fore.RED + f"{func} failed with error: {e}")  # Print error to console
        failures += 1

def main():
    global failures
    start = time.time()

    test_func(example)

    if failures == 0:
        print(Fore.GREEN + f"All tests succeeded. Time taken: {time.time() - start:.2f}s")
    else:
        print(Fore.RED + f"{failures} tests failed.") 

main()