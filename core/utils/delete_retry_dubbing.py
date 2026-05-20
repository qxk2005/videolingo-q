import os
import shutil

def delete_dubbing_files():
    files_to_delete = [
        os.path.join("output", "dub.wav"),
        os.path.join("output", "output_dub.mp4"),
        os.path.join("output", "audio", "tts_tasks.xlsx"),
    ]
    
    for file_path in files_to_delete:
        if os.path.exists(file_path):
            try:
                os.remove(file_path)
                print(f"Deleted: {file_path}")
            except Exception as e:
                print(f"Error deleting {file_path}: {str(e)}")
        else:
            print(f"File not found: {file_path}")
    
    for folder_name in ["segs", "tmp"]:
        folder = os.path.join("output", "audio", folder_name)
        if os.path.exists(folder):
            try:
                shutil.rmtree(folder)
                print(f"Deleted folder and contents: {folder}")
            except Exception as e:
                print(f"Error deleting folder {folder}: {str(e)}")
        else:
            print(f"Folder not found: {folder}")

if __name__ == "__main__":
    delete_dubbing_files()