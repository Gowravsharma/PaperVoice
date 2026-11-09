"""
upload_to_hf.py - Upload Kokoro models, voices, and operators to Hugging Face repository.
Repo: Vibrant8849/PaperVoice
"""
import sys
import pathlib
from huggingface_hub import HfApi

REPO_ID = "Vibrant8849/PaperVoice"

FILES_TO_UPLOAD = [
    ("models/kokoro/voices-v1.0.bin", "kokoro/voices-v1.0.bin"),
    ("models/kokoro/kokoro-v1.0.onnx", "kokoro/kokoro-v1.0.onnx"),
    ("models/kokoro/kokoro-v1.0.fast.onnx", "kokoro/kokoro-v1.0.fast.onnx"),
    ("models/kokoro/kokoro-v1.0.extreme.onnx", "kokoro/kokoro-v1.0.extreme.onnx"),
    ("models/kokoro/kokoro-v1.0.fused_snake.onnx", "kokoro/kokoro-v1.0.fused_snake.onnx"),
    ("tools/snake_op.dll", "kokoro/snake_op.dll"),
]

def main():
    api = HfApi()
    user_info = api.whoami()
    username = user_info.get("name")
    print(f"Logged in as Hugging Face user: {username}")
    print(f"Target repository: {REPO_ID}\n")

    for local_path_str, repo_path in FILES_TO_UPLOAD:
        local_path = pathlib.Path(local_path_str)
        if not local_path.exists():
            print(f"ERROR: Local file {local_path} does not exist. Skipping.")
            continue
        size_mb = local_path.stat().st_size / (1024 * 1024)
        print(f"Uploading {local_path} ({size_mb:.1f} MB) -> {repo_path} ...", flush=True)
        try:
            api.upload_file(
                path_or_fileobj=str(local_path),
                path_in_repo=repo_path,
                repo_id=REPO_ID,
                repo_type="model",
                commit_message=f"Add {repo_path} ({size_mb:.1f} MB)",
            )
            print(f"SUCCESS: {repo_path} uploaded successfully!\n", flush=True)
        except Exception as e:
            print(f"FAILED to upload {repo_path}: {e}\n", flush=True)

    print("All uploads completed!")

if __name__ == "__main__":
    main()
