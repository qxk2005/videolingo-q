import os
import shutil

def delete_dubbing_files():
    """Completely delete all dubbing-related files and folders for a clean start."""
    # Files in the root of output
    files_to_delete = [
        os.path.join("output", "dub.wav"),
        os.path.join("output", "dub.mp3"),
        os.path.join("output", "output_dub.mp4"),
        os.path.join("output", "normalized_dub.wav"),
    ]
    
    for file_path in files_to_delete:
        if os.path.exists(file_path):
            try:
                os.remove(file_path)
                print(f"Deleted: {file_path}")
            except Exception as e:
                print(f"Error deleting {file_path}: {str(e)}")

    # Entire audio folder (includes refers, segs, tmp, tasks, and mp3s)
    audio_folder = os.path.join("output", "audio")
    if os.path.exists(audio_folder):
        try:
            shutil.rmtree(audio_folder)
            print(f"Deleted entire audio folder: {audio_folder}")
        except Exception as e:
            print(f"Error deleting folder {audio_folder}: {str(e)}")
    else:
        print(f"Audio folder not found: {audio_folder}")

if __name__ == "__main__":
    delete_dubbing_files()