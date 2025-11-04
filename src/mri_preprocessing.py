import os
import nibabel as nib
import numpy as np
import cv2

# Base directories
RAW_DIR = "data/raw/"
PROCESSED_DIR = "data/processed/"

def preprocess_mri(file_path, output_dir=PROCESSED_DIR, target_size=(128, 128)):
    """
    Load .nii, .nii.gz, or .hdr/.img file, extract middle slice, normalize, resize, and save as .png
    """
    try:
        img = nib.load(file_path)
        data = img.get_fdata()

        # Take middle slice along the z-axis
        middle_slice = data[:, :, data.shape[2] // 2]

        # Normalize intensity (0–255)
        normalized = cv2.normalize(middle_slice, None, 0, 255, cv2.NORM_MINMAX)
        normalized = normalized.astype(np.uint8)

        # Resize to a standard size (128x128)
        resized = cv2.resize(normalized, target_size)

        # Save as PNG
        os.makedirs(output_dir, exist_ok=True)
        file_name = os.path.basename(file_path).split('.')[0] + ".png"
        output_path = os.path.join(output_dir, file_name)
        cv2.imwrite(output_path, resized)

        print(f"✅ Processed: {file_name}")
    except Exception as e:
        print(f"❌ Error processing {file_path}: {e}")


def process_all_mri(base_dir=RAW_DIR):
    """
    Recursively search all subfolders for MRI files (.nii, .nii.gz, .hdr)
    """
    for root, _, files in os.walk(base_dir):
        for file in files:
            if file.endswith((".nii", ".nii.gz", ".hdr")):
                full_path = os.path.join(root, file)
                preprocess_mri(full_path)


if __name__ == "__main__":
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    process_all_mri()
