import os
import nibabel as nib
import numpy as np
from PIL import Image

def process_mri_folder(input_dir, output_dir):
    """
    Converts MRI volumes (.hdr/.img or .nii) in input_dir into 2D PNG slices.
    """
    os.makedirs(output_dir, exist_ok=True)
    print(f"🧠 Processing MRI files from: {input_dir}")

    for file in os.listdir(input_dir):
        if file.endswith(".hdr") or file.endswith(".nii") or file.endswith(".nii.gz"):
            file_path = os.path.join(input_dir, file)
            print(f"📂 Reading volume: {file_path}")

            # Load MRI volume
            img = nib.load(file_path)
            data = img.get_fdata()

            # Normalize to 0–255
            data = (data - np.min(data)) / (np.max(data) - np.min(data)) * 255.0
            data = data.astype(np.uint8)

            # Save middle slices (1 out of every 5)
            middle_slices = range(data.shape[2] // 4, 3 * data.shape[2] // 4, 5)
            base_name = os.path.splitext(file)[0]

            for i in middle_slices:
                slice_img = data[:, :, i]

                # Ensure 2D
                if slice_img.ndim > 2:
                    slice_img = np.squeeze(slice_img)

                # Convert to grayscale image safely
                im = Image.fromarray(slice_img).convert("L")

                slice_path = os.path.join(output_dir, f"{base_name}_slice_{i}.png")
                im.save(slice_path)

            print(f"✅ Saved slices for {file}")

    print(f"\n🎉 All files processed successfully! Output folder: {output_dir}")
