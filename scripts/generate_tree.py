import os

IGNORE_FOLDERS = {
    "venv", "venv310", "__pycache__", "node_modules",
    "site-packages", ".git", "clusters", "raw_npz",
    "samples", "cnn_dataset", "predicted_masks"
}

IGNORE_EXTENSIONS = (".npz", ".csv")

OUTPUT_FILE = "project_structure.txt"


def generate_tree(path, prefix=""):
    lines = []

    items = sorted(os.listdir(path))

    for i, item in enumerate(items):
        full_path = os.path.join(path, item)

        connector = "└── " if i == len(items) - 1 else "├── "

        # show ignored folder but don't go inside
        if os.path.isdir(full_path) and item in IGNORE_FOLDERS:
            lines.append(prefix + connector + item)
            continue

        # skip unwanted file types
        if os.path.isfile(full_path) and item.lower().endswith(IGNORE_EXTENSIONS):
            continue

        lines.append(prefix + connector + item)

        if os.path.isdir(full_path):
            extension = "    " if i == len(items) - 1 else "│   "
            lines.extend(generate_tree(full_path, prefix + extension))

    return lines


tree = ["project"]
tree.extend(generate_tree("."))

with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    f.write("\n".join(tree))

print("✅ Clean project structure saved to", OUTPUT_FILE)