import sys
import os
import time
from colorama import init, Fore
init(autoreset=True)
from datasets import load_dataset

# Add the src directory to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src/easy_text_clustering')))

from clusterer import ClusterClassifier

failures = 0

def optimize_with_sample():
    texts = load_dataset('billingsmoore/text-clustering-example-data', split="train")['text']
    cc = ClusterClassifier(summary_create=False)
    cc.optimize_fit(texts, 2, sample_size=50)

optimize_with_sample()