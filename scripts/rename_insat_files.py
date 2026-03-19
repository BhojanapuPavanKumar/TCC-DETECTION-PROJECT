import os
import glob
from datetime import datetime

# Folder containing NPZ files
DATA_FOLDER = "data/samples"


def convert_filename(old_name):

    parts = old_name.split("_")

    # Skip if filename format is unexpected
    if len(parts) < 3:
        return None

    date_part = parts[1]   # 11FEB2025
    time_part = parts[2]   # 0015

    try:
        dt = datetime.strptime(date_part + time_part, "%d%b%Y%H%M")
    except:
        return None

    new_name = dt.strftime("%Y%m%d_%H%M") + ".npz"

    return new_name


def rename_files():

    files = glob.glob(os.path.join(DATA_FOLDER, "*.npz"))

    print("Total files:", len(files))

    renamed = 0
    skipped = 0

    for file_path in files:

        old_name = os.path.basename(file_path)

        # Skip if already renamed
        if old_name[:8].isdigit():
            skipped += 1
            continue

        new_name = convert_filename(old_name)

        if new_name is None:
            print("⚠ Skipping invalid:", old_name)
            skipped += 1
            continue

        new_path = os.path.join(DATA_FOLDER, new_name)

        # Skip if file already exists
        if os.path.exists(new_path):
            print("⚠ Already exists:", new_name)
            skipped += 1
            continue

        os.rename(file_path, new_path)
        renamed += 1

    print("\nRenamed:", renamed)
    print("Skipped:", skipped)


if __name__ == "__main__":
    rename_files()