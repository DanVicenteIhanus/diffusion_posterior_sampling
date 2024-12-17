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
    parser = argparse.ArgumentParser(description="Sort files in multiple folders by name and copy them to new folders.")
    parser.add_argument("parent_folder_path", type=str, help="Path to the parent folder containing folders to be sorted")
    args = parser.parse_args()

    try:
        parent_folder = args.parent_folder_path
        folders = [os.path.join(parent_folder, f) for f in os.listdir(parent_folder) if os.path.isdir(os.path.join(parent_folder, f))]
        
        if not folders:
            print("No subfolders found in the specified parent folder.")

        for folder in folders:
            print(f"Processing folder: {folder}")
            sort_folder_by_name(folder)

    except Exception as e:
        print(f"An unexpected error occurred while processing folders: {e}")

