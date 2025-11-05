import os
from src import mri_preprocessing

input_path = "data/raw/test_sample"
output_path = "data/processed/test_output"

print("🧠 Starting preprocessing test...")
mri_preprocessing.process_mri_folder(input_path, output_path)

if os.path.exists(output_path):
    pngs = [f for f in os.listdir(output_path) if f.endswith(".png")]
    print(f"✅ {len(pngs)} PNG slices created successfully.")
    print("Example:", pngs[:5])
else:
    print("❌ No output found. Check your preprocessing function.")
