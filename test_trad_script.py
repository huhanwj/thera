import os
import time
import glob
import numpy as np
from PIL import Image
import math # For ceiling function if needed, round is usually sufficient

# --- Configuration ---

# Input Directory containing low-resolution images (e.g., 360p)
INPUT_DIR = 'images/270p' # <--- SAME AS YOUR THERA SCRIPT INPUT

# Base Output Directory for traditional method results
BASE_OUTPUT_DIR = 'results_traditional' # <--- NEW base output directory

# Super-Resolution Target Scale (Fixed at 2x for this script)
SCALE_FACTOR = 4.0

# List of traditional methods to test (using Pillow's resampling filters)
METHODS = {
    "BICUBIC": Image.Resampling.BICUBIC,
    "LANCZOS": Image.Resampling.LANCZOS,
    # Add other Pillow filters here if desired, e.g.:
    "BILINEAR": Image.Resampling.BILINEAR,
    # "NEAREST": Image.Resampling.NEAREST_NEIGHBOR,
}

# Processing Options
WARMUP_COUNT = 3 # Number of images to process for warm-up before timing

# --- Main Processing Logic ---

# Sanity check input directory
if not os.path.isdir(INPUT_DIR):
    print(f"Error: Input directory '{INPUT_DIR}' not found.")
    print("Please create it and place your low-resolution images inside.")
    exit()

# Find input images
image_files = sorted(glob.glob(os.path.join(INPUT_DIR, '*.jpg'))) + \
              sorted(glob.glob(os.path.join(INPUT_DIR, '*.png')))

if not image_files:
    print(f"Error: No '.jpg' or '.png' images found in '{INPUT_DIR}'.")
    exit()

print(f"Found {len(image_files)} images in '{INPUT_DIR}'.")

# Create base output directory
os.makedirs(BASE_OUTPUT_DIR, exist_ok=True)

# --- Method Loop ---
for method_name, resampling_filter in METHODS.items():
    print("=" * 70)
    print(f"Starting processing for method: {method_name} (Scale: {SCALE_FACTOR:.1f}x)")
    print("=" * 70)

    # --- Prepare Output Directory for this Method ---
    # Include scale in subdir name even though it's fixed, for consistency
    output_subdir_name = f"{method_name}_scale_{SCALE_FACTOR:.1f}"
    output_dir = os.path.join(BASE_OUTPUT_DIR, output_subdir_name)
    os.makedirs(output_dir, exist_ok=True)
    print(f"Output directory for this run: '{output_dir}'")

    # --- Reset Timers and Process Images for this Method ---
    processing_times = []
    total_start_time_method = time.time() # Timer for this specific method run

    for i, img_path in enumerate(image_files):
        img_filename = os.path.basename(img_path)
        output_path = os.path.join(output_dir, img_filename)
        print(f"\n[{method_name} | Scale {SCALE_FACTOR:.1f}x] Processing image {i+1}/{len(image_files)}: '{img_filename}'")

        # --- Load Input Image ---
        try:
            # Load with Pillow and ensure RGB format
            img_pil = Image.open(img_path).convert('RGB')
            source_w, source_h = img_pil.size
            # print(f"Input image loaded: {source_w}x{source_h}")
        except FileNotFoundError:
            print(f"Error: Input image file not found at '{img_path}'. Skipping.")
            continue
        except Exception as e:
            print(f"Error loading image '{img_filename}': {e}. Skipping.")
            continue

        # --- Determine Target Shape ---
        target_w = round(source_w * SCALE_FACTOR)
        target_h = round(source_h * SCALE_FACTOR)
        target_shape = (target_w, target_h) # Pillow uses (W, H)
        # print(f"Target shape: {target_w}x{target_h}")

        # --- Run Upscaling ---
        start_time_img = time.time()
        try:
            # Perform the resize operation using the selected filter
            output_image_pil = img_pil.resize(target_shape, resample=resampling_filter)

        except Exception as e:
            print(f"Error during processing image '{img_filename}' (Method: {method_name}): {e}")
            continue # Skip saving and timing this image

        end_time_img = time.time()
        duration_img = end_time_img - start_time_img

        # --- Timing and Warm-up ---
        if i >= WARMUP_COUNT:
            processing_times.append(duration_img)
            print(f"Processing time: {duration_img:.5f} seconds") # More precision for fast methods
        else:
            print(f"Warm-up run {i+1}/{WARMUP_COUNT}. Time: {duration_img:.5f} seconds (not counted)")

        # --- Save Output Image ---
        try:
            output_image_pil.save(output_path)
            # print(f"Output image saved to '{output_path}'")
        except Exception as e:
            print(f"Error saving image '{output_path}': {e}")

    # --- Print Summary for the Current Method ---
    total_end_time_method = time.time()
    total_duration_method = total_end_time_method - total_start_time_method
    print("-" * 60)
    print(f"Finished processing for Method: {method_name}, Scale: {SCALE_FACTOR:.1f}x")
    print(f"Total time for this method (incl. warm-up): {total_duration_method:.2f} seconds")

    if processing_times:
        avg_time = sum(processing_times) / len(processing_times)
        print(f"Average processing time per image (after {WARMUP_COUNT} warm-ups): {avg_time:.5f} seconds")
        images_per_sec = 1.0 / avg_time if avg_time > 0 else float('inf')
        print(f"Average throughput: {images_per_sec:.2f} images/second")
    else:
        if len(image_files) > WARMUP_COUNT:
             print("No images were processed successfully after the warm-up phase to calculate average time.")
        else:
             print(f"Not enough images ({len(image_files)}) processed for warm-up ({WARMUP_COUNT}) and timing.")
    print("-" * 60)

print("\nTraditional upscaling comparison complete.")