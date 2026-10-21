"""Execute a generated notebook's code cells in order inside __main__ (so multiprocessing/fork can pickle the cell functions)."""
import json, sys
nb = json.load(open(sys.argv[1]))
for c in nb["cells"]:
    if c["cell_type"] != "code": continue
    name = c["metadata"].get("name", "?"); print(f"\n#### cell: {name}", flush=True)
    exec(compile("".join(c["source"]), f"<{name}>", "exec"), globals())
