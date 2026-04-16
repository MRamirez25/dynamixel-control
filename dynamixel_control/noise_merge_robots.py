#%%
from PIL import Image
from rembg import remove
import numpy as np
#%%
dir_path = "/home/mariano/Videos/noise_gripper_videos/"
img1 = Image.open(dir_path + "robot2_nobg.png").convert("RGBA")
img2 = Image.open(dir_path + "cup2_nobg.png").convert("RGBA")
#%%
# Convert to NumPy array and create mask for non-transparent pixels
img2_arr = np.array(img2)
mask = img2_arr[:, :, 3] > 0   # alpha > 0

# Crop to content
# coords = np.argwhere(mask)
# y0, x0 = coords.min(axis=0)
# y1, x1 = coords.max(axis=0)
# img2 = img2.crop((x0, y0, x1+1, y1+1))

# # Shift
# shift_x = 0   # right
# shift_y = 0  # up
# shifted_img2 = Image.new("RGBA", img1.size, (0, 0, 0, 0))
# shifted_img2.paste(img2, (shift_x, shift_y), img2)

# # --- Alpha-aware blending ---
bg_arr = np.array(img1).astype(np.float32)
fg_arr = np.array(img2).astype(np.float32)

# Opacity factor (like Google Docs)
opacity = 0.45
dy, dx = 200, -250
fg_arr = np.roll(fg_arr, shift=(dy, dx), axis=(0, 1))

# Mask of non-transparent pixels in foreground
fg_mask = fg_arr[:, :, 3] > int(255*0.9)

# Blend RGB channels only where mask is True
for c in range(3):  # R, G, B
    bg_arr[:, :, c][fg_mask] = (
        fg_arr[:, :, c][fg_mask] * opacity +
        bg_arr[:, :, c][fg_mask] * (1 - opacity)
    )

# Set alpha channel to fully opaque in the blended areas
bg_arr[:, :, 3][fg_mask] = 255

# Convert back to PIL Image and save
merged = Image.fromarray(bg_arr.astype(np.uint8), "RGBA")
merged.save("robot_merged.png")
#%%
img2 = remove(img2)
img2_arr = np.array(img2)
mask = img2_arr[:, :, 3] > 0   # alpha > 0
#%%
coords = np.argwhere(mask)
y0, x0 = coords.min(axis=0)
y1, x1 = coords.max(axis=0)

img2 = img2.crop((x0, y0, x1+1, y1+1))
#%%
img2_arr = np.array(img2)
#%%
# --- SHIFT IMAGE 2 ---
shift_x = 400   # positive = move right
shift_y = -400  # negative = move up

shifted_img2 = Image.new("RGBA", img1.size, (0, 0, 0, 0))
shifted_img2.paste(img2, (shift_x, shift_y), img2)

# --- MERGE WITH BLEND ---
merged = Image.blend(img1, shifted_img2, alpha=0.5)
merged.save("merged_shifted.png")
# %%
# result = Image.new("RGBA", img1.size, (0, 0, 0, 0))
# result.paste(img1, (0, 0), img1)
# opacity = 0.5
# img2.putalpha(int(255 * opacity))
# result.paste(img2, (shift_x, shift_y), img2)
# result.save("final.png")
# %%
threshold_value = int(255 * 0.9)
alpha = img2.split()[3]
alpha_mask = alpha.point(lambda p: 255 if p > threshold_value else 0)

# Reduce opacity of img2 where alpha > threshold
opacity_factor = 0.5 # 50% opacity
img2_arr = img2.copy()
img2_arr.putalpha(alpha.point(lambda p: int(p * opacity_factor) if p > threshold_value else 0))
img1.paste(img2_arr, (shift_x, shift_y), mask=alpha_mask)
img1.save("result.png")
# %%

arr = np.array(img2)
opacity_factor = 0.5
mask = arr[:, :, 3] > threshold_value
arr[:, :, 3][mask] = (arr[:, :, 3][mask] * opacity_factor).astype(np.uint8)
img2_modified = Image.fromarray(arr)
img1.paste(img2_modified, (shift_x, shift_y))
img1.save("result.png")

# %%

# Convert to NumPy array and create mask for non-transparent pixels
img2_arr = np.array(img2)
mask = img2_arr[:, :, 3] > 0   # alpha > 0

# Crop to content
coords = np.argwhere(mask)
y0, x0 = coords.min(axis=0)
y1, x1 = coords.max(axis=0)
img2 = img2.crop((x0, y0, x1+1, y1+1))

# Shift
shift_x = 0   # right
shift_y = 0  # up
shifted_img2 = Image.new("RGBA", img1.size, (0, 0, 0, 0))
shifted_img2.paste(img2, (shift_x, shift_y), img2)

# --- Alpha-aware blending ---
bg_arr = np.array(img1).astype(np.float32)
fg_arr = np.array(img2).astype(np.float32)

# Opacity factor (like Google Docs)
opacity = 0.3

# Mask of non-transparent pixels in foreground
fg_mask = fg_arr[:, :, 3] > int(255*0.9)

# Blend RGB channels only where mask is True
for c in range(3):  # R, G, B
    bg_arr[:, :, c][fg_mask] = (
        fg_arr[:, :, c][fg_mask] * opacity +
        bg_arr[:, :, c][fg_mask] * (1 - opacity)
    )

# Set alpha channel to fully opaque in the blended areas
bg_arr[:, :, 3][fg_mask] = 255

# Convert back to PIL Image and save
merged = Image.fromarray(bg_arr.astype(np.uint8), "RGBA")
merged = remove(merged,
    alpha_matting=True,
    alpha_matting_foreground_threshold=240,  # default ~240
    alpha_matting_background_threshold=10,   # default ~10
    alpha_matting_erode_size=10               # default ~10
)
merged.save("merged_robots.png")
# %%
#%%
dir_path = "/home/mariano/Videos/noise_gripper_videos/"
img1 = Image.open(dir_path + "robot1_nobg.png").convert("RGBA")
img2 = Image.open(dir_path + "cup2_nobg.png").convert("RGBA")
alpha=0.3
# Blend images
blended = Image.blend(img1, img2, alpha)

# Save result
blended.save("blended.png")
# %%
def overlay_blend(img1_path, img2_path, alpha=0.8, save_path="overlay_blend.png"):
    """
    Blend two images in "overlay" style so both images remain mostly visible.
    
    Args:
        img1_path (str): Path to base image
        img2_path (str): Path to top image
        alpha (float): Blend factor for img2 (0-1). 1 = fully img2, 0 = fully img1
        save_path (str): Path to save blended image
    """
    
    # Load images and convert to RGBA
    img1 = Image.open(img1_path).convert("RGBA")
    img2 = Image.open(img2_path).convert("RGBA")
    
    # Resize img2 to match img1 if needed
    if img1.size != img2.size:
        img2 = img2.resize(img1.size, Image.LANCZOS)
    
    # Convert to NumPy arrays
    arr1 = np.array(img1).astype(np.float32)
    arr2 = np.array(img2).astype(np.float32)
    
    # Overlay-style blending: weighted combination
    blended_rgb = arr1[:, :, :3] * (1 - alpha) + arr2[:, :, :3] * alpha
    
    # Alpha channel: take max to preserve transparency
    blended_alpha = np.maximum(arr1[:, :, 3], arr2[:, :, 3])
    
    # Combine RGB + alpha
    blended_arr = np.dstack((blended_rgb, blended_alpha))
    
    # Convert back to image and save
    blended_img = Image.fromarray(np.clip(blended_arr, 0, 255).astype(np.uint8), "RGBA")
    blended_img.save(save_path)
    print(f"Overlay blend saved to {save_path}")
    return blended_img

# Example usage
overlay_blend(f"{dir_path}robot1_nobg.png", f"{dir_path}cup2_nobg.png", alpha=0.7, save_path="overlay_result.png")
# %%
