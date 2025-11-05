import nibabel as nib
import matplotlib.pyplot as plt
import numpy as np
import os

# pick one of the hdr files you moved
hdr_path = "data/raw/test_sample/OAS1_0001_MR1_mpr-1_anon.hdr"

# make sure the file exists
if not os.path.exists(hdr_path):
    raise FileNotFoundError(f"File not found: {hdr_path}")

# load the MRI volume
img = nib.load(hdr_path)
data = img.get_fdata()
print(f"✅ Volume loaded successfully!\nShape: {data.shape}, dtype: {data.dtype}")

# show the middle slice
slice_index = data.shape[2] // 2
plt.imshow(data[:, :, slice_index], cmap="gray")
plt.title("Middle slice from MRI volume")
plt.axis("off")
plt.show()
