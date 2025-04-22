import os
import glob
import numpy as np
from PIL import Image
import pandas as pd
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
import torch
import lpips # Make sure lpips is installed (pip install lpips)
import time
import warnings # To potentially suppress warnings if needed, though seeing them is good

# --- Configuration ---

GROUND_TRUTH_DIR = 'images/1080p'
# THERA_RESULTS_BASE = 'results'
TRADITIONAL_RESULTS_BASE = 'results_traditional'
SPAN_RESULT_BASE = 'results_SPAN'
SCALE_SUFFIX = 'scale_4.0' # The specific scale subdirectory we're evaluating

# Models/Methods to evaluate
# THERA_VARIANTS = [
#     'edsr-air', 'rdn-air',
#     'edsr-plus', 'rdn-plus',
#     'edsr-pro', 'rdn-pro'
# ]
TRADITIONAL_METHODS = ['BICUBIC', 'LANCZOS', 'BILINEAR']
SPAN_Var = ['SPANX4_CH48', 'SPANX4_CH52']
# LPIPS Configuration
LPIPS_NETS = ['alex', 'vgg']
USE_GPU = torch.cuda.is_available()
DEVICE = torch.device("cuda" if USE_GPU else "cpu")
print(f"Using device: {DEVICE}")

# --- Helper Functions ---

def load_image_numpy(path):
    """Loads an image, converts to RGB, and returns as NumPy array [0, 1]."""
    try:
        img = Image.open(path).convert('RGB')
        img_np = np.array(img, dtype=np.float32) / 255.0
        # Basic sanity check for image size
        if img_np.shape[0] < 3 or img_np.shape[1] < 3:
             print(f"Warning: Image loaded from {path} has very small dimensions: {img_np.shape}")
             # Decide if you want to return None or the small image
             # return None
        return img_np
    except FileNotFoundError:
        print(f"Warning: File not found: {path}")
        return None
    except Exception as e:
        print(f"Warning: Could not load image {path}: {e}")
        return None

def numpy_to_lpips_tensor(img_np):
    """Converts NumPy HWC [0, 1] to PyTorch CHW [-1, 1] tensor."""
    tensor = torch.from_numpy(img_np.transpose((2, 0, 1))).float()
    tensor = (tensor * 2) - 1
    return tensor.unsqueeze(0).to(DEVICE)

# --- Initialization ---

lpips_models = {}
try:
    for net in LPIPS_NETS:
        print(f"Initializing LPIPS model with net='{net}'...")
        lpips_models[net] = lpips.LPIPS(net=net).to(DEVICE).eval()
    print("LPIPS models initialized.")
except Exception as e:
    print(f"FATAL ERROR: Could not initialize LPIPS models: {e}")
    exit()

gt_files = sorted(glob.glob(os.path.join(GROUND_TRUTH_DIR, '*.jpg'))) + \
           sorted(glob.glob(os.path.join(GROUND_TRUTH_DIR, '*.png')))

if not gt_files:
    print(f"Error: No ground truth images found in '{GROUND_TRUTH_DIR}'.")
    exit()
print(f"Found {len(gt_files)} ground truth images.")

all_results_dirs = {}
# for variant in THERA_VARIANTS:
#     model_key = f"THERA_{variant.upper()}"
#     dir_path = os.path.join(THERA_RESULTS_BASE, f"prs-eth-thera-{variant}", SCALE_SUFFIX)
#     if os.path.isdir(dir_path):
#         all_results_dirs[model_key] = dir_path
#     else:
#         print(f"Warning: Thera result directory not found, skipping: {dir_path}")

for method in TRADITIONAL_METHODS:
    method_key = method
    dir_path = os.path.join(TRADITIONAL_RESULTS_BASE, f"{method}_{SCALE_SUFFIX}")
    if os.path.isdir(dir_path):
        all_results_dirs[method_key] = dir_path
    else:
        print(f"Warning: Traditional result directory not found, skipping: {dir_path}")
for var in SPAN_Var:
    model_key = var
    dir_path = os.path.join(SPAN_RESULT_BASE, f"{var}", SCALE_SUFFIX)
    if os.path.isdir(dir_path):
        all_results_dirs[model_key] = dir_path
    else:
        print(f"Warning: SPAN result directory not found, skipping: {dir_path}")

if not all_results_dirs:
    print("Error: No result directories found to evaluate.")
    exit()
print("\nFound the following result directories to evaluate:")
for key, path in all_results_dirs.items():
    print(f"- {key}: {path}")

# --- Evaluation Loop ---
evaluation_data = []

for model_key, result_dir in all_results_dirs.items():
    print("\n" + "=" * 70)
    print(f"Evaluating: {model_key}")
    print("=" * 70)

    model_metrics = {
        'PSNR': [],
        'SSIM': [],
        'MAE': [] # <<< ADDED: List to store Mean Absolute Error values
    }
    for net in LPIPS_NETS:
        model_metrics[f'LPIPS_{net.upper()}'] = []

    image_count = 0
    processed_image_filenames = [] # Keep track of successfully processed images
    start_time_model = time.time()

    for gt_path in gt_files:
        gt_filename = os.path.basename(gt_path)
        result_path = os.path.join(result_dir, gt_filename)

        img_gt_np = load_image_numpy(gt_path)
        img_res_np = load_image_numpy(result_path)

        if img_gt_np is None or img_res_np is None:
            print(f"Skipping {gt_filename} for {model_key} due to loading error.")
            continue

        if img_gt_np.shape != img_res_np.shape:
            print(f"Warning: Dimension mismatch for {gt_filename}!")
            print(f"  GT shape: {img_gt_np.shape}, Result shape: {img_res_np.shape}")
            print(f"  Skipping metrics calculation for this image pair.")
            continue

        # Add this filename to the list *before* potential metric errors
        processed_image_filenames.append(gt_filename)
        current_metrics = {} # Store metrics for this image pair

        # --- Calculate Metrics ---
        try:
            # <<< ADDED: Calculate Mean Absolute Error (MAE) >>>
            # Calculates the mean of the absolute differences between pixels
            # Range: [0, 1] for float images. Lower is better.
            current_metrics['MAE'] = np.mean(np.abs(img_gt_np - img_res_np))

            # PSNR
            current_metrics['PSNR'] = psnr(img_gt_np, img_res_np, data_range=1.0)

            # --- Dynamic SSIM Calculation ---
            min_dim = min(img_res_np.shape[0], img_res_np.shape[1])
            # Default skimage win_size is 7 for SSIM. Let's check against that.
            if min_dim < 7:
                # If dimensions are too small, calculate the largest possible odd window size
                dynamic_win_size = min_dim if min_dim % 2 != 0 else min_dim - 1
                if dynamic_win_size < 3: # SSIM is ill-defined for win_size < 3
                    print(f"  Warning: Image {gt_filename} dimensions ({img_res_np.shape[0]}x{img_res_np.shape[1]}) too small for SSIM. Skipping SSIM calculation.")
                    current_metrics['SSIM'] = np.nan # Assign NaN if skipped
                else:
                    print(f"  Info: Adjusting SSIM win_size to {dynamic_win_size} for {gt_filename} due to small dimensions.")
                    # Use multichannel=True if RGB (shape has 3 dims and last is 3 or 4)
                    is_multichannel = img_res_np.ndim == 3 and img_res_np.shape[-1] in [3, 4]
                    current_metrics['SSIM'] = ssim(img_gt_np, img_res_np,
                                                   data_range=1.0,
                                                   win_size=dynamic_win_size,
                                                   channel_axis=-1 if is_multichannel else None) # Use channel_axis instead of multichannel if >=skimage 0.19
                                                   # multichannel=is_multichannel) # Use this for older skimage
            else:
                # Dimensions are large enough, use default or slightly smaller explicit size like 7 or 11
                is_multichannel = img_res_np.ndim == 3 and img_res_np.shape[-1] in [3, 4]
                current_metrics['SSIM'] = ssim(img_gt_np, img_res_np,
                                               data_range=1.0,
                                               win_size=11, # Explicitly use 11 (common default) or 7
                                               channel_axis=-1 if is_multichannel else None) # Use channel_axis instead of multichannel if >=skimage 0.19
                                               # multichannel=is_multichannel) # Use this for older skimage

            # LPIPS
            img_gt_tensor = numpy_to_lpips_tensor(img_gt_np)
            img_res_tensor = numpy_to_lpips_tensor(img_res_np)
            with torch.no_grad():
                for net in LPIPS_NETS:
                    lpips_model = lpips_models[net]
                    # Ensure tensors are on the same device as the model
                    current_metrics[f'LPIPS_{net.upper()}'] = lpips_model(img_gt_tensor.to(DEVICE), img_res_tensor.to(DEVICE)).item()

            # Append valid metrics
            for key, value in current_metrics.items():
                 if not np.isnan(value): # Only append if not NaN (i.e., calculation succeeded and wasn't skipped)
                     model_metrics[key].append(value) # <<< MODIFIED: Automatically appends MAE too

            image_count += 1
            if image_count % 20 == 0: # Print progress less often
                 print(f"  Processed {image_count}/{len(gt_files)} image pairs...")

        except Exception as e:
            print(f"Error calculating metrics for {gt_filename} (Model: {model_key}): {e}")
            # Note: Metrics for this image won't be appended if an error occurs here

    end_time_model = time.time()
    # Use len(processed_image_filenames) for reporting total attempted pairs
    print(f"Finished evaluating. Attempted {len(processed_image_filenames)} image pairs for {model_key}.")
    print(f"Successfully calculated metrics for {image_count} pairs in {end_time_model - start_time_model:.2f} seconds.")


    # --- Calculate Average Metrics for this Model ---
    # Averages are calculated only on successfully processed images for each metric
    # <<< MODIFIED: The existing loop automatically handles the new 'MAE' metric
    avg_metrics = {'Model': model_key, 'Num_Images_Avg': 0}
    non_empty_metrics = 0
    for metric_name, values in model_metrics.items():
        if values: # Check if list has valid results
            avg_metrics[f'Avg_{metric_name}'] = sum(values) / len(values)
            # Use the count from the first valid metric list found
            if non_empty_metrics == 0:
                avg_metrics['Num_Images_Avg'] = len(values)
            non_empty_metrics += 1
        else:
            avg_metrics[f'Avg_{metric_name}'] = np.nan # Assign NaN if no valid values

    # If absolutely no metrics were calculated for any image, Num_Images_Avg will be 0
    if non_empty_metrics == 0 and len(processed_image_filenames) > 0 :
         print(f"Warning: No metrics were successfully calculated for any image pair for {model_key}.")
         # Still add the row with NaNs
         avg_metrics['Num_Images_Avg'] = 0


    evaluation_data.append(avg_metrics)


# --- Final Report ---
print("\n" + "=" * 70)
print("Evaluation Summary")
print("=" * 70)

if evaluation_data:
    df_results = pd.DataFrame(evaluation_data)
    df_results = df_results.set_index('Model')

    # <<< MODIFIED: Add Avg_MAE to the desired column order
    # Decide where you want MAE - maybe next to LPIPS as it's an error metric (lower is better)
    cols_order = ['Avg_PSNR', 'Avg_SSIM', 'Avg_MAE'] + \
                 [f'Avg_LPIPS_{net.upper()}' for net in LPIPS_NETS] + \
                 ['Num_Images_Avg']
    # Ensure all columns exist, filling with NaN if one wasn't calculated for any model
    for col in cols_order:
        if col not in df_results.columns:
            df_results[col] = np.nan
    df_results = df_results[cols_order] # Reorder columns

    # Sort by PSNR (or change to MAE if preferred: sort_values(by='Avg_MAE', ascending=True))
    df_results = df_results.sort_values(by='Avg_PSNR', ascending=False)

    # <<< MODIFIED: Adjust float format if needed for MAE (4 decimal places is probably fine)
    pd.set_option('display.float_format', '{:.4f}'.format)
    print(df_results)

    # <<< MODIFIED: Update output filename if desired, e.g., to reflect scale or date
    # Example: Using SCALE_SUFFIX in the filename
    scale_num = SCALE_SUFFIX.split('_')[-1]
    output_csv_path = f"evaluation_summary_scale_{scale_num}.csv"
    try:
        df_results.to_csv(output_csv_path)
        print(f"\nResults saved to: {output_csv_path}")
    except Exception as e:
        print(f"\nError saving results to CSV: {e}")

else:
    print("No evaluation results were generated.")

print("\nEvaluation script finished.")