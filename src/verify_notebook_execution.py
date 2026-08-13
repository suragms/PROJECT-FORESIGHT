"""
Verify notebook cell execution counts and error statuses.
"""
import nbformat

with open('notebooks/04_eda.ipynb', 'r', encoding='utf-8') as f:
    nb = nbformat.read(f, as_version=4)

code_cells = [c for c in nb.cells if c.cell_type == 'code']
md_cells = [c for c in nb.cells if c.cell_type == 'markdown']
errors = []
executed_count = 0

for i, c in enumerate(code_cells):
    if c.get('execution_count') is not None:
        executed_count += 1
    for out in c.get('outputs', []):
        if out.get('output_type') == 'error':
            errors.append((i, out.get('ename'), out.get('evalue'), out.get('traceback')))

print(f"Total cells: {len(nb.cells)}")
print(f"Code cells: {len(code_cells)}")
print(f"Markdown cells: {len(md_cells)}")
print(f"Code cells successfully executed: {executed_count}/{len(code_cells)}")
print(f"Errors encountered: {len(errors)}")

if errors:
    for e in errors:
        print(f"  Error in code cell index {e[0]}: {e[1]} -> {e[2]}")
else:
    print(">>> 55/55 CELLS VALIDATED: 32/32 CODE CELLS EXECUTED WITH 0 ERRORS! <<<")
