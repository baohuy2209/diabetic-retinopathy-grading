# %% [markdown]
# # Prepare the pooled image and metadata splits

# %%
from pathlib import Path
import os
from IPython.display import display

# Edit these defaults in Colab, or set the DR_* environment variables before running Python.
IN_COLAB = 'COLAB_RELEASE_TAG' in os.environ or 'COLAB_BACKEND_VERSION' in os.environ

if IN_COLAB:
    import importlib.util
    import subprocess
    import sys
    dependencies = {'numpy': 'numpy', 'pandas': 'pandas', 'sklearn': 'scikit-learn', 'scipy': 'scipy', 'tqdm': 'tqdm', 'matplotlib': 'matplotlib', 'cv2': 'opencv-python-headless', 'albumentations': 'albumentations', 'PIL': 'pillow'}
    missing = [package for module, package in dependencies.items() if importlib.util.find_spec(module) is None]
    if missing:
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', *missing])

if IN_COLAB and not Path('/content/drive/MyDrive').is_dir():
    from google.colab import drive
    drive.mount('/content/drive')
THESIS_ROOT = Path(os.environ.get('DR_ROOT', '/content/drive/MyDrive/dr_thesis' if IN_COLAB else Path.cwd())).resolve()
INPUTS = Path(os.environ.get('DR_INPUTS', THESIS_ROOT / 'inputs')).resolve()
OUTPUTS = Path(os.environ.get('DR_OUTPUTS', THESIS_ROOT / 'outputs')).resolve()
WORK = Path(os.environ.get('DR_WORK', '/content/dr_thesis_work' if IN_COLAB else THESIS_ROOT / 'work')).resolve()
RAW = Path(os.environ.get('DR_RAW', INPUTS / 'raw'))
HYBRID = Path(os.environ.get('DR_HYBRID', OUTPUTS / 'prepared' / 'hybrid'))
FOLD_INPUT = Path(os.environ.get('DR_FOLDS', INPUTS / 'lodo_folds'))
IMAGE_INPUT = Path(os.environ.get('DR_IMAGES', INPUTS / 'native_cropped.zip' if
    (INPUTS / 'native_cropped.zip').is_file() else HYBRID / 'native_cropped'))
IDRID = Path(os.environ.get('DR_IDRID', INPUTS / 'idrid_segmentation'))

# %%
import os
import re
import random
import warnings
import numpy as np
import pandas as pd
import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import albumentations as A
from PIL import Image
from sklearn.model_selection import GroupShuffleSplit
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
# Suppress non-critical warnings
warnings.filterwarnings("ignore")

# Random seeds for research reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)

print(f"[INFO] Initialized preprocessing pipeline with SEED={SEED}")

# %% [markdown]
# ## 1. Dynamic Path Resolution & Configuration Dictionary

# %%
BASE_RAW_DIR = str(RAW)
BASE_PROCESSED_DIR = str(OUTPUTS / 'prepared')
HYBRID_RAW_REF = str(HYBRID)
NATIVE_PROCESSED_DIR = os.path.join(HYBRID_RAW_REF, "native_cropped")
os.makedirs(HYBRID_RAW_REF, exist_ok=True)
os.makedirs(NATIVE_PROCESSED_DIR, exist_ok=True)
print(f"[INFO] Raw Dir: {BASE_RAW_DIR}")
# Raw dataset configuration dictionary
DATASET_CONFIG = {
    "APTOS": {
        "name": "APTOS 2019",
        "csv_path": os.path.join("aptos", "aptos_total.csv"),
        "img_dir": os.path.join("aptos", "aptos_total_images"),
        "id_col": "id_code",
        "label_col": "diagnosis",
        "id_suffix": ".png"
    },
    "DDR": {
        "name": "DDR Dataset",
        "csv_path": os.path.join("ddr", "DR_grading.csv"),
        "img_dir": os.path.join("ddr", "DR_grading", "DR_grading"),
        "id_col": "id_code",
        "label_col": "diagnosis",
        "id_suffix": ".jpg"
    },
    "DeepDRiD": {
        "name": "DeepDRiD",
        "csv_path": os.path.join("deepdrid", "deepdrid_total.csv"),
        "img_dir": os.path.join("deepdrid", "deepdrid_total_images"),
        "id_col": "image_id",
        "label_col": "patient_DR_Level",
        "id_suffix": ".jpg"
    },
    "IDRiD": {
        "name": "IDRiD",
        "csv_path": os.path.join("idrid", "idrid_total.csv"),
        "img_dir": os.path.join("idrid", "idrid_total_images"),
        "id_col": "image_id",
        "label_col": "diagnosis",
        "id_suffix": ".jpg"
    },
    "Messidor-2": {
        "name": "Messidor-2",
        "csv_path": os.path.join("messidor2preprocess", "messidor_data.csv"),
        "img_dir": os.path.join("messidor2preprocess", "messidor-2", "messidor-2", "preprocess"),
        "id_col": "id_code",
        "label_col": "diagnosis",
        "id_suffix": ".png"
    },
    "EyePACS": {
        "name": "EyePACS (Kaggle)",
        "csv_path": os.path.join("zipEyepacs", "trainLabels.csv", "trainLabels.csv"),
        "img_dir": os.path.join("zipEyepacs", "train"),
        "id_col": "image",
        "label_col": "level",
        "id_suffix": ".jpeg"
    }
}

# %% [markdown]
# ## 2. Step 1: Master Dataset Harmonization & Schema Standardization
# 
# **Clinical Justification:** All 6 datasets map to the International Clinical Diabetic Retinopathy Severity Scale (ICDRS 0-4).
# Merging multi-center datasets exposes the network to diverse camera sensors (Topcon, Canon, Zeiss) and illumination conditions, improving cross-domain generalizability.

# %%
def parse_patient_and_side(image_id: str, dataset_key: str) -> tuple[str, str]:
    """Parse the cohort-specific patient and eye identifiers and prefix the patient key with the cohort."""
    img_str = str(image_id).lower()
    k = dataset_key.lower()
    
    if "idrid" in k:

        base = img_str.replace(".jpg", "").replace(".png", "")
        pid = base.split("_")[-1]
    elif "eyepacs" in k or "deepdrid" in k:
        # EyePACS: '10_left' -> '10'; DeepDRiD: '104_l1' -> '104'
        pid = img_str.split("_")[0]
    elif "ddr" in k:

        parts = img_str.replace(".jpg", "").replace(".png", "").split("-")
        pid = "-".join(parts[:2]) if len(parts) >= 2 else img_str.split(".")[0]
    elif "messidor" in k:

        pid = img_str.split("_")[0]
    else:

        pid = img_str
        
    side = "left" if "left" in img_str or "_l" in img_str else \
           "right" if "right" in img_str or "_r" in img_str else "unknown"
    return f"{dataset_key}::{pid}", side

def load_and_standardize_single_dataset(key: str, cfg: dict, base_dir: str) -> pd.DataFrame:
    full_csv_path = os.path.join(base_dir, cfg["csv_path"])
    full_img_dir = os.path.join(base_dir, cfg["img_dir"])
    
    if not os.path.exists(full_csv_path):
        print(f"[WARNING] Skipping {key} - CSV not found: {full_csv_path}")
        return pd.DataFrame()
        
    raw_df = pd.read_csv(full_csv_path)
    label_col = cfg["label_col"]
    if label_col not in raw_df.columns:
        possible = [c for c in raw_df.columns if "dr" in c.lower() or "level" in c.lower() or "diagnosis" in c.lower()]
        if possible:
            label_col = possible[0]
            
    df = raw_df[[cfg["id_col"], label_col]].copy()
    df.columns = ["image_id", "diagnosis"]
    df["dataset_name"] = cfg["name"]
    df["dataset_key"] = key
    
    # Ensure correct suffix
    if cfg["id_suffix"]:
        df["image_file"] = df["image_id"].astype(str).apply(lambda x: x if x.endswith(cfg["id_suffix"]) else x + cfg["id_suffix"])
    else:
        df["image_file"] = df["image_id"].astype(str)
        
    df["raw_full_path"] = df["image_file"].apply(lambda f: os.path.join(full_img_dir, f))
    
    # Clean non-integer labels
    df["diagnosis"] = pd.to_numeric(df["diagnosis"], errors="coerce")
    df = df.dropna(subset=["diagnosis"]).copy()
    df["diagnosis"] = df["diagnosis"].astype(int)
    df = df[df["diagnosis"].isin([0, 1, 2, 3, 4])]
    

    parsed_metadata = df["image_id"].apply(lambda x: parse_patient_and_side(x, dataset_key=key))
    df["patient_id"] = [p[0] for p in parsed_metadata]
    df["side"] = [p[1] for p in parsed_metadata]
    return df

def merge_and_harmonize_all_datasets(config_dict: dict, base_dir: str) -> pd.DataFrame:
    dfs = []
    for key, cfg in config_dict.items():
        df_sub = load_and_standardize_single_dataset(key, cfg, base_dir)
        if not df_sub.empty:
            dfs.append(df_sub)
            print(f"[HARMONIZED] {cfg['name']}: {len(df_sub):,} records")
            
    master_df = pd.concat(dfs, ignore_index=True)
    print(f"\n[SUCCESS] Master Harmonized Dataset: {len(master_df):,} total images across {master_df['dataset_name'].nunique()} sources.")
    return master_df

master_df = merge_and_harmonize_all_datasets(DATASET_CONFIG, BASE_RAW_DIR)

# %%
print("=== UNIQUE PATIENT COUNTS BY DATASET ===")
patient_counts = master_df.groupby("dataset_name")["patient_id"].nunique()
print(patient_counts)
print("\nTotal unique patient keys:", master_df["patient_id"].nunique())
master_df

# %% [markdown]
# ## 4. Step 2 & Step 3: Geometric Preprocessing & Image Enhancement
# 
# **Auto-Crop ROI:** Thresholding technique to remove non-diagnostic black borders around the fundus circular region.

# %%
def crop_fundus_roi(img_bgr: np.ndarray, tolerance: int = 7) -> np.ndarray:
    """
    Auto-crops uninformative black borders around fundus photograph.
    """
    if img_bgr is None or img_bgr.size == 0:
        return img_bgr
        
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    mask = gray > tolerance
    
    if not np.any(mask):
        return img_bgr
        
    # Crop to bounding box of non-black pixels
    row_mask = np.any(mask, axis=1)
    col_mask = np.any(mask, axis=0)
    ymin, ymax = np.where(row_mask)[0][[0, -1]]
    xmin, xmax = np.where(col_mask)[0][[0, -1]]
    
    cropped = img_bgr[ymin:ymax+1, xmin:xmax+1]
    return cropped if cropped.size > 0 else img_bgr
def process_single_image(args):
    raw_path, out_path = args
    if os.path.exists(out_path):
        return True
    try:
        raw_img = cv2.imread(raw_path)
        if raw_img is None:
            return False
        cropped = crop_fundus_roi(raw_img)
        os.makedirs(os.path.dirname(out_path), exist_ok=True)

        cv2.imwrite(out_path, cropped, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
        return True
    except Exception:
        return False

# %% [markdown]
# ## 5. Step 4: Albumentations Augmentation & Targeted Training Set Balancing
# 
# **Balancing Strategy (Applied STRICTLY to 70% Training Set):**
# - Target Count: Exactly 2,000 images per DR severity grade (10,000 total training images).
# - Class 0, Class 1, Class 2: Randomly undersampled to 2,000 images.
# - Class 3, Class 4: Oversampled & augmented via Albumentations to 2,000 images.
# - Validation (8,097) & Test (8,097) sets: Left untouched in raw distribution.

# %%
print("\n>>> Prepare the native-cropped image store.")


master_df["native_path"] = master_df.apply(
    lambda r: os.path.join(
        NATIVE_PROCESSED_DIR, 
        r["dataset_key"], 
        re.sub(r'[^\w\.-]', '_', str(r['image_id'])) + '.jpg'
    ),
    axis=1
)


for key in master_df["dataset_key"].unique():
    os.makedirs(os.path.join(NATIVE_PROCESSED_DIR, key), exist_ok=True)


tasks = list(zip(master_df["raw_full_path"], master_df["native_path"]))


num_workers = min(8, os.cpu_count() or 4)
print(f"[INFO] Preprocessing with {num_workers} workers...")

with ThreadPoolExecutor(max_workers=num_workers) as executor:
    results = list(
        tqdm(
            executor.map(process_single_image, tasks),
            total=len(tasks),
            desc="Processing Native Images"
        )
    )


master_df = master_df[master_df["native_path"].apply(os.path.exists)].reset_index(drop=True)
print(f" Saved {len(master_df):,} images to the native-cropped store.")

# %%
# ==========================================

# ==========================================
def create_balanced_train_pool(train_df: pd.DataFrame, target_range=(1500, 2500), seed=42) -> pd.DataFrame:
    balanced_dfs = []
    for grade in range(5):
        grade_subset = train_df[train_df["diagnosis"] == grade].copy().reset_index(drop=True)
        current_count = len(grade_subset)
        if current_count == 0:
            continue
            
        datasets_in_grade = grade_subset["dataset_name"].unique()
        num_datasets = len(datasets_in_grade)
        
        if current_count > target_range[1]:
            target_count = int(np.random.RandomState(seed + grade).randint(target_range[0], target_range[1] + 1))
            base_quota = target_count // num_datasets
            sampled_datasets = []
            leftover_quota = 0
            eligible = []
            
            for ds_name in datasets_in_grade:
                ds_subset = grade_subset[grade_subset["dataset_name"] == ds_name]
                if len(ds_subset) <= base_quota:
                    sampled_datasets.append(ds_subset)
                    leftover_quota += (base_quota - len(ds_subset))
                else:
                    eligible.append((ds_name, ds_subset, len(ds_subset)))
                    
            if eligible:
                quota_large = base_quota + (leftover_quota // len(eligible))
                for ds_name, ds_subset, count in eligible:
                    sampled_datasets.append(ds_subset.sample(min(quota_large, count), random_state=seed))
                    
            selected = pd.concat(sampled_datasets, ignore_index=True)
            if len(selected) > target_count:
                selected = selected.sample(target_count, random_state=seed).reset_index(drop=True)
            elif len(selected) < target_count:
                rem = grade_subset[~grade_subset["native_path"].isin(selected["native_path"])]
                if not rem.empty:
                    extra = rem.sample(min(target_count - len(selected), len(rem)), random_state=seed)
                    selected = pd.concat([selected, extra], ignore_index=True)
            balanced_dfs.append(selected)
        else:
            balanced_dfs.append(grade_subset)
            
    return pd.concat(balanced_dfs, ignore_index=True)
def split_master_pool_patient_level(df: pd.DataFrame, random_state=42):
    gss_train = GroupShuffleSplit(n_splits=1, train_size=0.70, random_state=random_state)
    train_idx, temp_idx = next(gss_train.split(df, groups=df['patient_id']))
    
    train_df = df.iloc[train_idx].copy().reset_index(drop=True)
    temp_df = df.iloc[temp_idx].copy().reset_index(drop=True)
    
    gss_test = GroupShuffleSplit(n_splits=1, train_size=0.5, random_state=random_state)
    val_idx, test_idx = next(gss_test.split(temp_df, groups=temp_df['patient_id']))
    
    val_df = temp_df.iloc[val_idx].copy().reset_index(drop=True)
    test_df = temp_df.iloc[test_idx].copy().reset_index(drop=True)
    
    train_df["split"] = "train"
    val_df["split"] = "val"
    test_df["split"] = "test"
    return train_df, val_df, test_df

# ==========================================

# ==========================================

print(">>> Split patients into train, validation and test sets (70/15/15).")
train_raw_df, val_raw_df, test_raw_df = split_master_pool_patient_level(master_df, random_state=SEED)

print("\n>>> Balance classes and source cohorts in the training set.")
train_balanced_df = create_balanced_train_pool(train_raw_df, target_range=(1500, 2500), seed=SEED)

val_final_df = val_raw_df.copy()
test_final_df = test_raw_df.copy()


train_balanced_df.to_csv(os.path.join(HYBRID_RAW_REF, "train_metadata.csv"), index=False)
val_final_df.to_csv(os.path.join(HYBRID_RAW_REF, "val_metadata.csv"), index=False)
test_final_df.to_csv(os.path.join(HYBRID_RAW_REF, "test_metadata.csv"), index=False)

print("\n" + "="*50)
print("=== DATASET SUMMARY ===")
print("="*50)
print("Train Set (Balanced):\n", train_balanced_df["diagnosis"].value_counts().sort_index())
print("\nVal Set (Natural Prevalence):\n", val_final_df["diagnosis"].value_counts().sort_index())
print("\nTest Set (Natural Prevalence):\n", test_final_df["diagnosis"].value_counts().sort_index())
print("\nTest Prevalence (%):\n", (test_final_df["diagnosis"].value_counts(normalize=True).sort_index() * 100).round(2))

# %% [markdown]
# ## 6. Step 6: Pixel Normalization & Pipeline Verification

# %%
def normalize_image_tensor(img_np: np.ndarray, mode: str = "zscore") -> np.ndarray:
    """
    Normalizes pixel values to [0, 1] or standard ImageNet Z-score.
    """
    img_float = img_np.astype(np.float32) / 255.0
    if mode == "minmax":
        return img_float
    elif mode == "zscore":
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        return (img_float - mean) / std
    return img_float

# Save processed metadata tables
train_balanced_df.to_csv(os.path.join(HYBRID_RAW_REF, "train_balanced_metadata.csv"), index=False)
val_final_df.to_csv(os.path.join(HYBRID_RAW_REF, "val_raw_metadata.csv"), index=False)
test_final_df.to_csv(os.path.join(HYBRID_RAW_REF, "test_raw_metadata.csv"), index=False)

print("\n=== FINAL PREPROCESSING & BALANCING SUMMARY ===")
print("Balanced Train Set (512x512):")
print(train_balanced_df["diagnosis"].value_counts().sort_index())

print("\nUntouched Validation Set:")
print(val_final_df["diagnosis"].value_counts().sort_index())

print("\nUntouched Test Set:")
print(test_final_df["diagnosis"].value_counts().sort_index())
