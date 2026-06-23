import json

with open("graphrag_lab.ipynb", "r", encoding="utf-8") as f:
    nb = json.load(f)

with open("notebook_cells.txt", "w", encoding="utf-8") as out:
    for i, cell in enumerate(nb.get("cells", [])):
        ctype = cell.get("cell_type")
        source = "".join(cell.get("source", []))
        out.write(f"\n\n==========================================\n")
        out.write(f"CELL {i} - TYPE: {ctype.upper()}\n")
        out.write(f"==========================================\n")
        out.write(source)
        out.write("\n")

print("Saved cell contents to notebook_cells.txt")
