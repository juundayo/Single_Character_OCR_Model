# ----------------------------------------------------------------------------#

import os
import json
import matplotlib.pyplot as plt
from collections import defaultdict

# ----------------------------------------------------------------------------#

# Paths!
dataset_path = r"C:\Users\bgat\OCR\Full_Dataset"
mapping_file = os.path.join(dataset_path, "char_to_id.json")

# Loading character-to-ID mapping and inverting it.
with open(mapping_file, "r", encoding="utf-8") as f:
    char_to_id = json.load(f)
id_to_char = {str(v): k for k, v in char_to_id.items()}

# Counting the files per class folder.
class_counts = defaultdict(int)
for folder_name in os.listdir(dataset_path):
    folder_path = os.path.join(dataset_path, folder_name)
    if os.path.isdir(folder_path) and folder_name.isdigit():
        file_count = len([f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))])
        class_counts[folder_name] = file_count

# Data preparation.
sorted_ids = sorted(class_counts.keys(), key=lambda x: int(x))
labels = [id_to_char.get(i, f"Unknown({i})") for i in sorted_ids]
counts = [class_counts[i] for i in sorted_ids]

# Plotting!
plt.figure(figsize=(12, 6))
bars = plt.bar(labels, counts, color="#4C72B0", edgecolor="black")

# Adding value labels on top of bars.
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, height + 1, str(height), ha='center', va='bottom', fontsize=9)

# Styling!
plt.title("Class Distribution by Greek Character", fontsize=14, weight='bold')
plt.xlabel("Greek Character", fontsize=12)
plt.ylabel("Number of Files", fontsize=12)
plt.xticks(rotation=45, ha='right')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()

plt.show()
