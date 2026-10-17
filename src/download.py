from modelscope import snapshot_download

model_dir = snapshot_download(
    "Qwen/Qwen3-8B",
    cache_dir="/root/autodl-tmp/"
)
print(f"模型已下载到：{model_dir}")