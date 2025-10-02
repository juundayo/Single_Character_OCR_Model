# ----------------------------------------------------------------------------#

import os
import shutil
from pathlib import Path

# ----------------------------------------------------------------------------#

def organize_dataset():
    # Paths!
    dataset_path = r"C:\Users\bgat\OCR\Dataset"
    output_base = r"C:\Users\bgat\OCR"

    training_path = os.path.join(output_base, "Training")
    validation_path = os.path.join(output_base, "Validation") 
    testing_path = os.path.join(output_base, "Testing")
    
    # Defining splits based on the writer numbers in the filenames.
    splits = {
        "Training": range(27, 97),     # 027-096 (70 writers)
        "Validation": range(97, 112),  # 097-111 (15 writers)
        "Testing": range(112, 127)     # 112-126 (15 writers)
    }
    
    # Creating output dictionaries.
    for folder in [training_path, validation_path, testing_path]:
        os.makedirs(folder, exist_ok=True)
        print(f"Created directory: {folder}")
    
    # Getting all subfolders in the dataset.
    try:
        subfolders = [f for f in os.listdir(dataset_path) 
                     if os.path.isdir(os.path.join(dataset_path, f))]
        print(f"Found {len(subfolders)} subfolders in dataset")
    except FileNotFoundError:
        print(f"Error: Dataset path '{dataset_path}' not found!")
        return
    except PermissionError:
        print(f"Error: Permission denied accessing '{dataset_path}'!")
        return
    
    # First, create the same subfolder structure in all three split directories.
    for subfolder in subfolders:
        for split_dir in [training_path, validation_path, testing_path]:
            os.makedirs(os.path.join(split_dir, subfolder), exist_ok=True)
    
    print("Created subfolder structure in all split directories")
    
    # Now iterate through each subfolder and copy files to appropriate splits.
    for subfolder in subfolders:
        source_folder = os.path.join(dataset_path, subfolder)
        
        try:
            files = [f for f in os.listdir(source_folder) 
                    if os.path.isfile(os.path.join(source_folder, f))]
            
            print(f"\nProcessing folder '{subfolder}' with {len(files)} files")
            
            for filename in files:
                try:
                    # Extracting writer number from filename (first 3 digits)
                    # Example: "027_3_L_01_W_01_C_01.tif" -> 27
                    writer_str = filename.split('_')[0]
                    writer_num = int(writer_str)
                    
                    # Determining which split this file belongs to.
                    destination = None
                    for split_name, split_range in splits.items():
                        if writer_num in split_range:
                            destination = split_name
                            break
                    
                    if destination is None:
                        print(f"Warning: Writer {writer_num} not in any split range. Skipping file: {filename}")
                        continue
                    
                    # Defining source and destination paths.
                    source_file = os.path.join(source_folder, filename)
                    dest_folder = os.path.join(output_base, destination, subfolder)
                    dest_file = os.path.join(dest_folder, filename)
                    
                    # Copying the file to the new folder.
                    shutil.copy2(source_file, dest_file)
                    
                except ValueError:
                    print(f"Warning: Cannot extract writer number from filename: {filename}. Skipping.")
                except Exception as e:
                    print(f"Error processing file '{filename}': {str(e)}")
            
        except Exception as e:
            print(f"Error accessing folder '{subfolder}': {str(e)}")
    
    print("\nDataset organization completed!")

    print("\nSummary:")
    for split_name in splits.keys():
        split_path = os.path.join(output_base, split_name)
        if os.path.exists(split_path):
            total_files = 0
            for subfolder in subfolders:
                subfolder_path = os.path.join(split_path, subfolder)
                if os.path.exists(subfolder_path):
                    files_count = len([f for f in os.listdir(subfolder_path) 
                                     if os.path.isfile(os.path.join(subfolder_path, f))])
                    total_files += files_count
            print(f"{split_name}: {total_files} total files")

if __name__ == "__main__":
    organize_dataset()
