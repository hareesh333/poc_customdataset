import os
import re
 
def rename_files(folder_path, pdf_file_name):
    # Regular expression to match the current format
    pattern = re.compile(r'^page_(\d+)')
 
    # List all files in the folder
    for filename in os.listdir(folder_path):
        # Check if the file is a .pdf or .txt file
        if filename.endswith('.jpg') or filename.endswith('.txt'):
            # Match the file name with the pattern
            match = pattern.match(filename)
            if match:
                # Extract the page number
                page_number = match.group(1)
                # Construct the new file name
                new_name = f"page_{page_number}-{pdf_file_name}{os.path.splitext(filename)[1]}"
                # Define the full paths for renaming
                old_file = os.path.join(folder_path, filename)
                new_file = os.path.join(folder_path, new_name)
                # Rename the file
                os.rename(old_file, new_file)
                print(f"Renamed {filename} to {new_name}")
 
# Example usage
folder_path = '/home/harish/Documents/exportdata/txtfiles'
pdf_file_name = '357124_14129_BFK_1_1E_2019_10'  # This will be appended to the file names
rename_files(folder_path, pdf_file_name)
