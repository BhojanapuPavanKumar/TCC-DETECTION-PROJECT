import os
import re
import pkg_resources

PROJECT_DIR = "."
OUTPUT_FILE = "requirements.txt"

STANDARD_LIBS = {
    "os", "sys", "math", "time", "datetime", "random",
    "re", "glob", "json", "collections", "itertools",
    "subprocess", "threading", "multiprocessing",
    "pathlib", "shutil", "logging", "pickle", "gc"
}

PROJECT_MODULES = {
    "preprocessing", "tracking", "training", "utils",
    "dataset", "prediction", "detection", "lifecycle"
}

# Fix naming differences
LIBRARY_MAPPING = {
    "sklearn": "scikit-learn"
}


def extract_imports(file_path):
    imports = set()

    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()

            match1 = re.match(r"import\s+([a-zA-Z0-9_\.]+)", line)
            if match1:
                imports.add(match1.group(1).split(".")[0])

            match2 = re.match(r"from\s+([a-zA-Z0-9_\.]+)\s+import", line)
            if match2:
                imports.add(match2.group(1).split(".")[0])

    return imports


def scan_project():
    all_imports = set()

    for root, dirs, files in os.walk(PROJECT_DIR):

        if "venv" in root or "__pycache__" in root:
            continue

        for file in files:
            if file.endswith(".py"):
                path = os.path.join(root, file)
                all_imports.update(extract_imports(path))

    return all_imports


def get_version(package_name):
    try:
        return pkg_resources.get_distribution(package_name).version
    except:
        return None


def clean_and_resolve(libs):
    final_libs = []

    for lib in libs:

        if lib in STANDARD_LIBS or lib in PROJECT_MODULES:
            continue

        lib = LIBRARY_MAPPING.get(lib, lib)

        version = get_version(lib)

        if version:
            final_libs.append(f"{lib}=={version}")
        else:
            final_libs.append(lib)

    return sorted(set(final_libs))


def save_requirements(libs):
    with open(OUTPUT_FILE, "w") as f:
        for lib in libs:
            f.write(lib + "\n")


if __name__ == "__main__":
    print("Scanning project...")
    libs = scan_project()

    print("Resolving versions...")
    libs = clean_and_resolve(libs)

    print("Saving requirements.txt...")
    save_requirements(libs)

    print("\n✅ requirements.txt with versions generated!")