import os
import random
import shutil

main_folder = "/ibex/user/shiekhmf/sada/overlaid-3"
train_folder = "/ibex/user/shiekhmf/sada/overlaid-split/train"
valid_folder = "/ibex/user/shiekhmf/sada/overlaid-split/valid"
test_folder = "/ibex/user/shiekhmf/sada/overlaid-split/test"

train_ratio = 0.75  # Adjust according to your desired ratio
valid_ratio = 0.15
test_ratio = 0.1

# Create folders if they don't exist
for folder in [train_folder, valid_folder, test_folder]:
    os.makedirs(folder, exist_ok=True)

# List all files/subfolders in the main folder
items = os.listdir(main_folder)
random.shuffle(items)

# Calculate split indices
total_items = len(items)
train_split = int(total_items * train_ratio)
valid_split = int(total_items * (train_ratio + valid_ratio))

# Move or copy files to respective folders
for i, item in enumerate(items):
    src_path = os.path.join(main_folder, item)
    if i < train_split:
        dst_path = os.path.join(train_folder, item)
    elif i < valid_split:
        dst_path = os.path.join(valid_folder, item)
    else:
        dst_path = os.path.join(test_folder, item)
    shutil.copy(src_path, dst_path)  # Use shutil.copy if you want to copy instead

