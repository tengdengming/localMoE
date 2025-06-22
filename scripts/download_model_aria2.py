#!/usr/bin/env python3
import os
import argparse
from huggingface_hub import HfApi
import subprocess

def main():
    parser = argparse.ArgumentParser(description="使用 aria2 从 Hugging Face 自动下载模型全部文件")
    parser.add_argument("repo_id", type=str, help="模型仓库 ID，例如 Qwen/Qwen3-30B-A3B")
    parser.add_argument("--token", type=str, default=os.environ.get("HF_TOKEN"), help="Hugging Face 访问令牌")
    parser.add_argument("--out-dir", type=str, default=None, help="下载输出目录，默认为当前目录下的模型名")
    parser.add_argument("--no-accelerate", action="store_true", help="禁用 aria2 多线程加速")

    args = parser.parse_args()
    repo_id = args.repo_id
    hf_token = args.token
    out_dir = args.out_dir or f"./{repo_id.replace('/', '_')}"
    use_accelerate = not args.no_accelerate

    if not hf_token:
        print("? 需要 Hugging Face Token，请使用 --token 参数或设置环境变量 HF_TOKEN")
        return

    print(f"?? 模型仓库: {repo_id}")
    print(f"?? 下载目录: {out_dir}")
    print(f"?? 使用 Token: {'已提供' if hf_token else '未提供'}")

    os.makedirs(out_dir, exist_ok=True)
    aria2_list_path = os.path.join(out_dir, "aria2_download_list.txt")

    print("?? 获取模型文件列表...")
    api = HfApi()
    files = api.list_repo_files(repo_id, repo_type="model", token=hf_token)

    print(f"?? 共 {len(files)} 个文件，生成下载任务列表...")

    with open(aria2_list_path, "w") as f:
        for file in files:
            url = f"https://huggingface.co/{repo_id}/resolve/main/{file}"
            f.write(f"{url}\n")
            f.write(f"  dir={out_dir}\n")
            f.write(f"  out={file}\n")
            f.write(f"  header=Authorization: Bearer {hf_token}\n\n")

    print("? 下载任务列表生成完毕，开始下载...")

    aria2_args = [
        "aria2c",
        "--input-file", aria2_list_path,
        "--continue",
        "--auto-file-renaming=false",
        "--summary-interval=5"
    ]

    if use_accelerate:
        aria2_args += [
            "-x", "16",
            "-s", "16",
            "--max-connection-per-server=16"
        ]

    subprocess.run(aria2_args)

    print("?? 所有文件下载完成！")

if __name__ == "__main__":
    main()
