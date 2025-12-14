# import numpy as np

# try:
#     data = np.load('face_templates.npz')
#     print("File loaded successfully!")
# except Exception as e:
#     print(f"Error loading file: {e}")
import zipfile

try:
    with zipfile.ZipFile('face_templates.npz', 'r') as zip_ref:
        zip_ref.testzip()  # Check if the archive is valid
        zip_ref.extractall('extracted_contents')  # Extract all files
    print("Extraction successful.")
except zipfile.BadZipFile as e:
    print(f"Error: {e}")
