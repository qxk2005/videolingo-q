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

    # Specific folders and files inside audio folder to delete
    audio_folder = os.path.join("output", "audio")
    if os.path.exists(audio_folder):
        sub_items = os.listdir(audio_folder)
        for item in sub_items:
            # Preserve essential source files and subtitle files
            if item in [
                "raw.mp3", 
                "vocal.mp3", 
                "background.mp3", 
                "src_subs_for_audio.srt", 
                "trans_subs_for_audio.srt"
            ]:
                continue
            
            item_path = os.path.join(audio_folder, item)
            try:
                if os.path.isfile(item_path):
                    os.remove(item_path)
                elif os.path.isdir(item_path):
                    shutil.rmtree(item_path)
                print(f"Deleted: {item_path}")
            except Exception as e:
                print(f"Error deleting {item_path}: {str(e)}")
    else:
        print(f"Audio folder not found: {audio_folder}")

if __name__ == "__main__":
    delete_dubbing_files()