import os
import shutil
import argparse

def sort_folder_by_name(folder_path):

    try:
        items = os.listdir(folder_path)
        files = [f for f in items if os.path.isfile(os.path.join(folder_path, f))]
        sorted_files = sorted(files)
        sorted_folder_name = os.path.basename(folder_path) + "_sorted"
        sorted_folder_path = os.path.join(os.path.dirname(folder_path), sorted_folder_name)
        os.makedirs(sorted_folder_path, exist_ok=True)

        for file in sorted_files:
            src = os.path.join(folder_path, file)
            dst = os.path.join(sorted_folder_path, file)
            shutil.copy2(src, dst)

        print(f"Files have been copied to the new folder: {sorted_folder_path}")

    except FileNotFoundError:
        print(f"Error: The folder '{folder_path}' does not exist.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sort files in a folder by name and copy them to a new folder.")
    parser.add_argument("folder_path", type=str, help="Path to the folder to be sorted")
    args = parser.parse_args()

    sort_folder_by_name(args.folder_path)
