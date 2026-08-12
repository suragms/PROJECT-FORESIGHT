"""
Execute 04_eda.ipynb and record cell outputs.
"""
import os
import sys
import time
import nbformat
from nbconvert.preprocessors import ExecutePreprocessor

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NB_PATH = os.path.join(BASE_DIR, "notebooks", "04_eda.ipynb")

print(f"Loading notebook: {NB_PATH}")
with open(NB_PATH, "r", encoding="utf-8") as f:
    nb = nbformat.read(f, as_version=4)

ep = ExecutePreprocessor(timeout=600, kernel_name="python3")

print(f"Executing {len(nb.cells)} cells...")
t0 = time.time()
ep.preprocess(nb, {"metadata": {"path": BASE_DIR}})
t1 = time.time()
print(f"Notebook executed successfully in {t1 - t0:.2f} seconds.")

with open(NB_PATH, "w", encoding="utf-8") as f:
    nbformat.write(nb, f)

print(f"Executed notebook saved to {NB_PATH}")
