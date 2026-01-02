#!/usr/bin/env python3
"""
Hugging Face Spaces へファイルをアップロードするスクリプト

使い方:
  export HF_TOKEN="your_huggingface_token"
  python3 upload_to_hf.py
"""
from huggingface_hub import HfApi, login
import os
import sys

# 認証情報（環境変数から取得）
HF_TOKEN = os.environ.get("HF_TOKEN")
if not HF_TOKEN:
    print("❌ Error: HF_TOKEN environment variable is not set")
    print("Please set it with: export HF_TOKEN='your_token'")
    sys.exit(1)

REPO_ID = "jin3141/Pinn-test"
REPO_TYPE = "space"

# ログイン
login(token=HF_TOKEN)

# APIクライアント作成
api = HfApi()

# アップロードするファイルのリスト
files_to_upload = [
    "app.py",
    "pinn_burgers.py",
    "numerical_solver.py",
    "requirements.txt",
    "README.md",
    "Dockerfile",
    ".gitignore"
]

print(f"🚀 Uploading files to {REPO_ID}...")

# 各ファイルをアップロード
for file_path in files_to_upload:
    if os.path.exists(file_path):
        print(f"  📤 Uploading {file_path}...")
        try:
            api.upload_file(
                path_or_fileobj=file_path,
                path_in_repo=file_path,
                repo_id=REPO_ID,
                repo_type=REPO_TYPE,
                token=HF_TOKEN
            )
            print(f"  ✅ {file_path} uploaded successfully")
        except Exception as e:
            print(f"  ❌ Failed to upload {file_path}: {e}")
    else:
        print(f"  ⚠️  {file_path} not found, skipping")

print(f"\n✨ Deployment complete!")
print(f"🌐 Your app will be available at: https://huggingface.co/spaces/{REPO_ID}")
print(f"⏳ Building may take 5-10 minutes...")
