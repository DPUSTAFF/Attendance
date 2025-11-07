# PageRank Algorithm Implementation using Python and BeautifulSoup

import requests
from bs4 import BeautifulSoup
import numpy as np

# -------------------------------
# Step 1: Define the web pages (sample URLs for demonstration)
# -------------------------------
pages = {
    "A": ["B", "C"],
    "B": ["C"],
    "C": ["A"],
    "D": ["C"]
}

# -------------------------------
# Step 2: Initialize parameters
# -------------------------------
num_pages = len(pages)
page_names = list(pages.keys())
damping_factor = 0.85   # probability of following a link
iterations = 100         # number of times to iterate
ranks = np.ones(num_pages) / num_pages  # start with equal rank

# -------------------------------
# Step 3: Build link matrix
# -------------------------------
M = np.zeros((num_pages, num_pages))

for i, p in enumerate(page_names):
    links = pages[p]
    if len(links) > 0:
        for l in links:
            j = page_names.index(l)
            M[j][i] = 1 / len(links)

# -------------------------------
# Step 4: Apply PageRank formula iteratively
# -------------------------------
for _ in range(iterations):
    ranks = (1 - damping_factor) / num_pages + damping_factor * M.dot(ranks)

# -------------------------------
# Step 5: Display results
# -------------------------------
print("\nFinal PageRank Scores:")
for i, page in enumerate(page_names):
    print(f"{page}: {ranks[i]:.4f}")