import os
import pickle
import time
import glob
import shutil # For renaming files

import jax
from jax import jit
import jax.numpy as jnp
# Check if GPU is available and configure JAX
try:
    print("JAX devices:", jax.devices())
    # Optional: uncomment to force CPU if needed for debugging
    # jax.config.update('jax_platform_name', 'cpu')
except Exception as e:
    print(f"Could not list JAX devices: {e}")

from chunkax import chunk
import numpy as np
from PIL import Image

# Assuming 'model.py' and 'utils.py' are in the same directory or accessible
# If not, adjust Python path if necessary
try:
    from model import build_thera
    from utils import make_grid, interpolate_grid
except ImportError as e:
    print(f"Error importing local modules (model.py, utils.py): {e}")
    print("Please ensure these files are in the same directory or your Python path.")
    exit()

from huggingface_hub import hf_hub_download # For downloading the model

# --- Configuration ---

# Input Directory containing low-resolution images (e.g., 144p)
INPUT_DIR = 'images/270p' # <--- CHANGE THIS if needed

# Base Output Directory for results
BASE_OUTPUT_DIR = 'results' # <--- CHANGE THIS if needed

# Ground Truth is disabled for this batch run
GROUND_TRUTH_BASE_DIR: str | None = None # Set to None

# Super-Resolution Target Scales to test
SCALING_FACTORS = [4.0] # <--- ADJUST AS NEEDED

# List of Hugging Face Model Repository IDs to test
MODEL_REPOS = [
    'prs-eth/thera-edsr-air',
    'prs-eth/thera-rdn-air',
    'prs-eth/thera-edsr-plus',
    'prs-eth/thera-rdn-plus',
    'prs-eth/thera-edsr-pro',
    'prs-eth/thera-rdn-pro',
] # <--- ADD or REMOVE models as needed

# Directory to save the downloaded model checkpoints
CHECKPOINT_DIR = './checkpoints'
HF_FILENAME = 'model.pkl' # Default filename on Hugging Face

# Processing Options
DO_ENSEMBLE = True # Use geometric self-ensemble (rotations)
PATCH_SIZE_ENC = 256 # Patch size for encoder (reduce if OOM)
PATCH_SIZE_DEC = 256 # Patch size for decoder (fixed in original script)
WARMUP_COUNT = 3 # Number of images to process for warm-up before timing

# --- Constants ---
MEAN = jnp.array([.4488, .4371, .4040], dtype=jnp.float32)
VAR = jnp.array([.25, .25, .25], dtype=jnp.float32) # stddev=0.5 -> variance=0.25
STD = jnp.sqrt(VAR)

# --- Helper Functions (ensure float32) ---

# --- Helper Functions (ensure float32) ---

def process_single(source, apply_encoder, apply_decoder, params, target_shape, patch_size_enc):
    """Processes a single image (or rotated version). Expects HWC float32 input."""
    # Calculate t based on target scale relative to source H
    target_h, target_w = target_shape # Unpack for clarity later
    source_h, source_w, source_c = source.shape # Unpack source shape
    scale_ratio = target_h / source_h
    t = jnp.float32(scale_ratio**-2)[None]

    # --- Debug Print ---
    # print(f"    [Debug process_single] Start. Source shape: {(source_h, source_w, source_c)}, Target shape: {(target_h, target_w)}")

    coords_nearest = jnp.asarray(make_grid(target_shape), dtype=jnp.float32)[None] # (1, H_tgt, W_tgt, 2)

    # source_chw = source.transpose(2, 0, 1) # HWC -> CHW
    source_up = interpolate_grid(coords_nearest, source[None]) # Expects (B, C, H_in, W_in) input -> (B, C, H_tgt, W_tgt) output
    # source_up = source_up.squeeze(0).transpose(1, 2, 0) # (1, C, H_tgt, W_tgt) -> (C, H_tgt, W_tgt) -> (H_tgt, W_tgt, C)

    # --- Debug Print ---
    # print(f"    [Debug process_single] source_up shape after interpolate/transpose: {source_up.shape} (Expected: ({target_h}, {target_w}, {source_c}))")
    # if source_up.shape != (target_h, target_w, source_c):
    #     print(f"    [Debug process_single] WARNING: source_up shape mismatch!")


    source_std = (source - MEAN) / STD
    source_std = source_std[None] # Add batch dim (B, H_in, W_in, C)

    if patch_size_enc is not None:
        apply_encoder_chunked = chunk(apply_encoder, patch_size_enc, (None, (-3, -2)))
    else:
        apply_encoder_chunked = apply_encoder

    encoding = apply_encoder_chunked(params, source_std)

    coords = jnp.asarray(make_grid(target_shape), dtype=jnp.float32)[None]  # (1, H_tgt, W_tgt, 2)

    apply_decoder_chunked = chunk(
        apply_decoder,
        PATCH_SIZE_DEC,
        (None, None, (-3, -2), None), # Chunk over coords' H, W dims
    )

    out = apply_decoder_chunked(params, encoding, coords, t) # Expects (B, H_tgt, W_tgt, C) outputs

    # --- Debug Print ---
    # print(f"    [Debug process_single] 'out' shape after decoder: {out.shape} (Expected: (1, {target_h}, {target_w}, {source_c}))")
    # if out.shape != (1, target_h, target_w, source_c):
    #      print(f"    [Debug process_single] WARNING: 'out' shape mismatch after decoder!")


    # De-standardize and add residual connection (source_up)
    out = out * STD[None, None, None, :] + MEAN[None, None, None, :]

    # --- vvvvv FINAL DEBUG AREA vvvvv ---
    # print(f"    [Debug process_single] Shape of 'out' before residual add: {out.shape}")
    # print(f"    [Debug process_single] Shape of 'source_up' before residual add: {source_up.shape}")
    # print(f"    [Debug process_single] Shape of 'source_up[None]' before residual add: {source_up[None].shape}")
    # --- ^^^^^ FINAL DEBUG AREA ^^^^^ ---

    out += source_up # Add batch dim to source_up (HWC -> B,H,W,C)


    # print(f"    [Debug process_single] Final 'out' shape: {out.shape}") # Optional: check after add
    return out


def process_batch(source_hwc, model, params, target_shape, do_ensemble=True, patch_size_enc=None):
    """Processes an image with optional ensembling. Expects HWC float32 input."""
    # Jit the apply functions once per model/scale processing start
    # Moved JIT outside the loop for efficiency if possible, but JIT per call is fine too
    apply_encoder = jit(model.apply_encoder)
    apply_decoder = jit(model.apply_decoder)

    outs = []
    for i_rot in range(4 if do_ensemble else 1):
        # Rotate HWC input: axes=(0, 1) corresponds to H and W axes
        source_ = jnp.rot90(source_hwc, k=i_rot, axes=(-3, -2))
        # Target shape needs swapping if rotated 90 or 270 degrees
        target_shape_ = tuple(reversed(target_shape)) if i_rot % 2 else target_shape

        # process_single expects HWC float32 input
        out_single_batch = process_single(
            source_, apply_encoder, apply_decoder, params, target_shape_, patch_size_enc)

        # Rotate back: axes=(1, 2) on the output (B, H, W, C)
        out_rotated_back = jnp.rot90(out_single_batch, k=-i_rot, axes=(1, 2)) # Rotate H, W axes back
        outs.append(out_rotated_back)

    # Stack results along a new dimension (axis 0), then mean reduce
    out_ensembled = jnp.mean(jnp.stack(outs), axis=0).clip(0., 1.)

    # Remove batch dimension and convert to uint8
    # Result from process_single has shape (1, H, W, C)
    out_final_np = jnp.rint(out_ensembled[0] * 255).astype(jnp.uint8)

    # Ensure output is on CPU as numpy array before returning
    return jax.device_get(out_final_np)


# --- Main Processing Logic ---

# Sanity check input directory
if not os.path.isdir(INPUT_DIR):
    print(f"Error: Input directory '{INPUT_DIR}' not found.")
    print("Please create it and place your low-resolution images inside.")
    exit()

# Find input images
image_files = sorted(glob.glob(os.path.join(INPUT_DIR, '*.jpg'))) + \
              sorted(glob.glob(os.path.join(INPUT_DIR, '*.png'))) # Add png support

if not image_files:
    print(f"Error: No '.jpg' or '.png' images found in '{INPUT_DIR}'.")
    exit()

print(f"Found {len(image_files)} images in '{INPUT_DIR}'.")

# Create base checkpoint directory
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(BASE_OUTPUT_DIR, exist_ok=True)

# --- Model Loop ---
for repo_id in MODEL_REPOS:
    print("=" * 70)
    print(f"Starting processing for model: {repo_id}")
    print("=" * 70)
    model_name_safe = repo_id.replace('/', '-') # Create a file-system safe name
    descriptive_checkpoint_filename = f"{model_name_safe}.pkl"
    checkpoint_path_renamed = os.path.join(CHECKPOINT_DIR, descriptive_checkpoint_filename)
    checkpoint_path_original = os.path.join(CHECKPOINT_DIR, HF_FILENAME)

    # --- Download and Rename Model ---
    if not os.path.exists(checkpoint_path_renamed):
        print(f"\nDownloading model '{HF_FILENAME}' from '{repo_id}'...")
        try:
            hf_hub_download(
                repo_id=repo_id,
                filename=HF_FILENAME,
                local_dir=CHECKPOINT_DIR,
                local_dir_use_symlinks=False
            )
            if os.path.exists(checkpoint_path_original):
                 # Ensure the target doesn't exist if rename fails mid-way previously
                 if os.path.exists(checkpoint_path_renamed):
                     os.remove(checkpoint_path_renamed)
                 shutil.move(checkpoint_path_original, checkpoint_path_renamed)
                 print(f"Model downloaded and saved as '{descriptive_checkpoint_filename}'")
            else:
                 if not os.path.exists(checkpoint_path_renamed):
                     raise FileNotFoundError(f"Downloaded file '{HF_FILENAME}' not found for renaming.")
                 else:
                     print(f"Model checkpoint '{descriptive_checkpoint_filename}' already exists (likely renamed previously).")
        except Exception as e:
            print(f"Error downloading or renaming model {repo_id}: {e}")
            print("Skipping this model.")
            continue
    else:
        print(f"\nUsing existing checkpoint: '{descriptive_checkpoint_filename}'")

    # --- Load Checkpoint and Build Model ---
    print(f"Loading checkpoint from '{checkpoint_path_renamed}'...")
    try:
        with open(checkpoint_path_renamed, 'rb') as fh:
            check = pickle.load(fh)
            params = check['model']
            # Optional: move params to device explicitly if needed, though JAX usually handles it
            # params = jax.device_put(params)
            backbone = check['backbone']
            size = check['size']
        print(f"Checkpoint loaded. Backbone: {backbone}, Size: {size}")

        print("Building Thera model...")
        model = build_thera(3, backbone, size)
        print("Model built successfully.")

    except FileNotFoundError:
        print(f"Error: Checkpoint file not found at '{checkpoint_path_renamed}'. Skipping model.")
        continue
    except NameError:
        print("Error: 'build_thera' function not defined. Is 'model.py' accessible?")
        continue
    except Exception as e:
        print(f"Error loading checkpoint or building model: {e}")
        print("Skipping this model.")
        continue

    # Create base output directory for this specific model
    model_output_base_dir = os.path.join(BASE_OUTPUT_DIR, model_name_safe)
    os.makedirs(model_output_base_dir, exist_ok=True)

    # --- Scale Factor Loop ---
    for scale in SCALING_FACTORS:
        print("-" * 60)
        print(f"Processing model '{model_name_safe}' with Scale Factor: {scale:.1f}")
        print("-" * 60)

        # --- Prepare Output Directory for this Model and Scale ---
        output_subdir_name = f"scale_{scale:.1f}"
        output_dir = os.path.join(model_output_base_dir, output_subdir_name)
        os.makedirs(output_dir, exist_ok=True)
        print(f"Output directory for this run: '{output_dir}'")

        # --- Reset Timers and Process Images for this Scale ---
        processing_times = []
        total_start_time_scale = time.time() # Timer for this specific scale run

        for i, img_path in enumerate(image_files):
            img_filename = os.path.basename(img_path)
            output_path = os.path.join(output_dir, img_filename)
            print(f"\n[{model_name_safe} | Scale {scale:.1f}] Processing image {i+1}/{len(image_files)}: '{img_filename}'")

            # --- Load Input Image ---
            try:
                img_pil = Image.open(img_path).convert('RGB')
                # Convert to NumPy HWC, float32, [0, 1] range
                source_np = np.asarray(img_pil, dtype=np.float32) / 255.0
                source_h, source_w = source_np.shape[:2]
                # print(f"Input image loaded: {source_w}x{source_h}") # Less verbose output
            except FileNotFoundError:
                print(f"Error: Input image file not found at '{img_path}'. Skipping.")
                continue
            except Exception as e:
                print(f"Error loading image '{img_filename}': {e}. Skipping.")
                continue

            # --- Determine Target Shape ---
            target_h = round(source_h * scale)
            target_w = round(source_w * scale)
            target_shape = (target_h, target_w) # H, W format
            # print(f"Target shape: {target_w}x{target_h}") # Less verbose output

             # Convert source numpy array to JAX array *once* per image load
            source_jax = jnp.asarray(source_np)

            # --- Run Super-Resolution ---
            start_time_img = time.time()
            try:
                # process_batch expects HWC float32 JAX input
                output_image_np_uint8 = process_batch(
                    source_jax,         # Shape (H_src, W_src, C) float32 JAX array
                    model,
                    params,
                    target_shape,       # Tuple (H_tgt, W_tgt)
                    do_ensemble=DO_ENSEMBLE,
                    patch_size_enc=PATCH_SIZE_ENC
                )
                # output_image_np_uint8 is now guaranteed NumPy uint8 HWC by process_batch

                # Ensure the JAX computation is finished before stopping timer
                jax.block_until_ready(output_image_np_uint8) # Important for accurate timing

            except Exception as e:
                print(f"Error during processing image '{img_filename}' (Model: {model_name_safe}, Scale: {scale:.1f}): {e}")
                if "ResourceExhaustedError" in str(e) or "OOM" in str(e):
                     print("Suggestion: OOM Error likely occurred. Try reducing PATCH_SIZE_ENC or processing smaller images.")
                # Potentially add more specific error handling if needed
                continue # Skip saving and timing this image

            end_time_img = time.time()
            duration_img = end_time_img - start_time_img

            # --- Timing and Warm-up ---
            if i >= WARMUP_COUNT:
                processing_times.append(duration_img)
                print(f"Processing time: {duration_img:.3f} seconds")
            else:
                print(f"Warm-up run {i+1}/{WARMUP_COUNT}. Time: {duration_img:.3f} seconds (not counted)")

            # --- Save Output Image ---
            try:
                output_image_pil = Image.fromarray(output_image_np_uint8) # Expects HWC uint8
                output_image_pil.save(output_path)
                # print(f"Output image saved to '{output_path}'") # Less verbose output
            except Exception as e:
                print(f"Error saving image '{output_path}': {e}")

            # --- Ground Truth Comparison Removed ---
            # The GT block is omitted as GROUND_TRUTH_BASE_DIR is None


        # --- Print Summary for the Current Model and Scale Factor ---
        total_end_time_scale = time.time()
        total_duration_scale = total_end_time_scale - total_start_time_scale
        print("-" * 60)
        print(f"Finished processing for Model: {model_name_safe}, Scale: {scale:.1f}")
        print(f"Total time for this scale (incl. warm-up): {total_duration_scale:.2f} seconds")

        if processing_times:
            avg_time = sum(processing_times) / len(processing_times)
            print(f"Average processing time per image (after {WARMUP_COUNT} warm-ups): {avg_time:.3f} seconds")
            images_per_sec = 1.0 / avg_time if avg_time > 0 else float('inf')
            print(f"Average throughput: {images_per_sec:.2f} images/second")
        else:
            if len(image_files) > WARMUP_COUNT:
                 print("No images were processed successfully after the warm-up phase to calculate average time.")
            else:
                 print(f"Not enough images ({len(image_files)}) processed for warm-up ({WARMUP_COUNT}) and timing.")
        print("-" * 60)


    print(f"\nFinished all scaling factors for model: {repo_id}\n")


print("\nBatch processing complete for all models and scaling factors.")