from huggingface_hub import HfApi

REPO_ID = "trl-lab/kaggleds-corpus-task-based-search-bench"
BASE = "evaluation/huggingface"

splits = {
    "train":      f"{BASE}/eval_data_train.csv",
    "validation": f"{BASE}/eval_data_validation.csv",
    "test":       f"{BASE}/eval_data_test.csv",
}

api = HfApi()

# Clean up any existing data folder (e.g. parquet files from failed attempts)
try:
    api.delete_folder(path_in_repo="data", repo_id=REPO_ID, repo_type="dataset")
    print("Cleaned up existing data folder.")
except Exception:
    pass

# Upload each split as a raw CSV — HF auto-detects splits from filenames
print("Uploading splits...")
for name, path in splits.items():
    print(f"  {name} — uploading...")
    api.upload_file(
        path_or_fileobj=path,
        path_in_repo=f"data/{name}.csv",
        repo_id=REPO_ID,
        repo_type="dataset",
    )
    print(f"  {name}: done.")

# Upload README.md from local file
print("\nUploading README.md...")
api.upload_file(
    path_or_fileobj=f"{BASE}/README.md",
    path_in_repo="README.md",
    repo_id=REPO_ID,
    repo_type="dataset",
)

print("\nAll done.")
