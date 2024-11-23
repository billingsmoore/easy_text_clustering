import sys
import os
import time
from colorama import init, Fore
init(autoreset=True)

# Add the src directory to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src/easy_text_clustering')))

from datasets import load_dataset
from optimizer import Optimizer
from clusterer import ClusterClassifier

texts = load_dataset('billingsmoore/text-clustering-example-data', split="train")['text']

failures = 0

def test_optimize_texts():
    global texts
    opt = Optimizer()
    umap_args, hdbscan_args = opt.fit(texts, 2, sample_size=1000)
    print(umap_args)
    print(hdbscan_args)

def test_optimize_embeddings():
    global texts
    opt=Optimizer()
    cc = ClusterClassifier()
    embeddings = cc.embed(texts)
    a, b = opt.fit(embeddings)
    print(a)
    print(b)

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

    test_func(test_optimize_texts)

    if failures == 0:
        print(Fore.GREEN + f"All tests succeeded. Time taken: {time.time() - start:.2f}s")
    else:
        print(Fore.RED + f"{failures} tests failed.") 

main()