# sft-with-chatweb

基于 **Qwen3-8B** + **QLoRA 4-bit**（Unsloth + TRL SFTTrainer）的中文医疗问答 Helper。

> 训练：在 6 个科室（内科 / 外科 / 儿科 / 妇产科 / 男科 / 肿瘤科）各 2 万条问答（共 12 万）上做 1 epoch 微调。
> 部署：`python run.py serve` 一键启动 OpenAI 兼容 API + Web 聊天界面（同端口），支持流式输出、计时、多轮会话、医疗免责声明。
> Qwen3 思考模式：推理默认关闭（`enable_thinking=False`），并对接输出做 `<think>` 段过滤，保证回答直接呈现。

---

## 目录结构

```
sft-with-chatweb/
├── run.py                             # 统一入口：data / train / merge / serve
├── 中文医疗数据集/                     # 中文医疗问答原始数据（6 科室，GBK CSV）
├── src/                               # 核心代码
│   ├── build_dataset.py               # CSV → JSONL 数据预处理
│   ├── train.py                       # Unsloth + LoRA 训练
│   ├── merge_lora.py                  # LoRA → 合并 HF 模型
│   ├── download.py                    # ModelScope 基座下载
│   ├── modelServer.py                 # OpenAI 兼容 API + Web 聊天界面托管（一键启动）
│   └── serve/
│       └── static/                    # 聊天前端
│           ├── index.html
│           ├── style.css
│           ├── app.js
│           └── avatar.jpg
├── data/                              # 处理后数据
│   └── processed/
│       ├── train.jsonl
│       ├── val.jsonl
│       ├── _dryrun.jsonl
│       └── stats.json
├── finetune/                          # 合并后的微调模型（.env 的 MERGED_MODEL_PATH）
├── outputs/                           # 训练产物
│   ├── lora_adapter/                  # LoRA 权重 ~100MB
│   └── logs/
├── requirements.txt
└── .env / .env.example
```

---

## 数据文件说明（data/processed）

| 文件 | 内容 | 生成方式 |
| --- | --- | --- |
| `train.jsonl` | 训练集，约占全部样本的 98% | `python run.py data` |
| `val.jsonl` | 验证集，约占 2% | `python run.py data` |
| `_dryrun.jsonl` | 训练集头部 200 条，仅供 dry-run 验证流水线 | `python run.py train --dry-run`（缺失时由 `run.py` 自动生成） |
| `stats.json` | 各科室清洗 / 采样统计与最终条数 | `python run.py data` |

### train.jsonl / val.jsonl / _dryrun.jsonl

三个文件格式完全一致：UTF-8 编码、一行一条 JSON（JSONL），每条只含 `messages` 字段，按 Qwen ChatML 的 `system → user → assistant` 三段组织；训练时由 `tokenizer.apply_chat_template` 套用具体模板。样例（为便于阅读做了缩进，实际文件中每条为一行）：

```json
{
  "messages": [
    {
      "role": "system",
      "content": "你是一名专业的中文医疗助手，请基于医学知识为患者提供准确、严谨的回答，并在必要时建议就医。"
    },
    {
      "role": "user",
      "content": "我得了癌症，一直在接受化疗，我想请问一下化疗药外渗的应急预案？"
    },
    {
      "role": "assistant",
      "content": "应立即停止化疗药物的输注，并报告医生和护士长。护士应及时了解化疗药物的名称，剂量，输注的方法，评估患者药物外渗的穿刺部位面积，外渗药物的量，皮肤的颜色，温度，疼痛的性质。护理人员准确评估外渗药液损失量，如损失量超过原药液量的10%，再重新输注时应遵医嘱不足损失量。出现化疗药物外渗时应立即遵医嘱作皮下封闭。"
    }
  ]
}
```

- `system`：固定医疗助手提示词（所有样本相同）
- `user`：患者问题
- `assistant`：医生回答
- 科室信息不写入单条记录，只在 `stats.json` 中按科室汇总
- 条数由 `--n-per-dept` 决定：默认每科 20000 条 → 约 117600 train / 2400 val；当前 `data/processed` 中的样例是用 `--n-per-dept 100` 构建的（588 / 12，`data/` 已 gitignore，不入库）

### stats.json

记录构建参数与逐科室统计，便于核对采样是否均衡。每科字段：`raw`（原始行数）、`after_length`（长度过滤后）、`after_dedup`（去重后）、`sampled`（实际采样数）、`avg_ask_chars` / `avg_answer_chars`（平均字符数）。样例（省略其余科室，实际文件含全部 6 个科室）：

```json
{
  "n_per_dept": 100,
  "max_char_len": 600,
  "seed": 42,
  "departments": {
    "内科": {
      "raw": 220602,
      "after_length": 216390,
      "after_dedup": 188112,
      "sampled": 100,
      "avg_ask_chars": 55.6,
      "avg_answer_chars": 145.5
    }
  },
  "total_after_dedup": 696318,
  "total_after_sample": 600,
  "train_size": 588,
  "val_size": 12
}
```

- `n_per_dept` / `max_char_len` / `seed`：本次构建使用的参数
- `departments`：6 个科室各自的清洗漏斗统计
- `total_after_dedup` / `total_after_sample`：全量去重后条数 / 最终采样条数
- `train_size` / `val_size`：写入两个 JSONL 的实际行数

---

## 快速开始（训练 3 步 + 一键部署）

> 推荐环境：Linux + GPU ≥ 24GB（RTX 3090 / 4090 / A5000 / A100）。

### 0. 创建 Conda 环境 + 安装依赖

**已验证环境：PyTorch 2.5.1 / Python 3.11 / RTX 3090 24GB（驱动 570 / CUDA 12.8，装 cu124 轮子）**

```bash
# 1) 创建并激活 conda 环境（Python 3.11 对 Unsloth / bitsandbytes 兼容性最稳）
conda create -n medical-sft python=3.11 -y
conda activate medical-sft

# 2) 安装 PyTorch（必须 cu124；兼容支持 CUDA 12.4+ 的驱动，如 12.6/12.8；勿装默认的 cu130）
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 \
    --index-url https://download.pytorch.org/whl/cu124

# 3) 安装项目依赖（会钉死 torch，避免被 unsloth 升级）
#    Qwen3-8B 最低要求 transformers>=4.51、unsloth>=2025.5（旧版无 Qwen3 架构）
pip install -r requirements.txt
# 若环境里已有 torchao，请卸掉（与 torch 2.5.1 不兼容）:
# pip uninstall -y torchao

# 4) 验证 GPU 可用
python -c "import torch; print('PyTorch:', torch.__version__, '| CUDA:', torch.version.cuda, '| device:', torch.cuda.get_device_name(0))"
# 预期: PyTorch: 2.5.1+cu124 | CUDA: 12.4 | device: NVIDIA GeForce RTX 3090

# 5) 配置环境变量（run.py 与 src 下各入口会自动加载 .env）
cp .env.example .env
# 编辑 .env，至少确认 BASE_MODEL_PATH 指向含 config.json 的 Qwen3-8B 基座目录
# 也可直接下载: python src/download.py （ModelScope，注意改其中的 cache_dir）
```

### 1. 构建数据集

```bash
# 原始数据放在 ./中文医疗数据集（6 个科室子目录，GBK CSV，已 gitignore）
# 路径可用 .env 的 RAW_DATA_DIR 覆盖
python run.py data
# 输出: data/processed/{train,val}.jsonl + stats.json
# 默认每科 20000 条，共 ~117600 train + ~2400 val
# 可选: python run.py data --n-per-dept 10000 --seed 42
```

### 2. 训练（**先 dry-run 再正式**）

```bash
# (a) Dry-run：用 200 条数据跑 10 步验证流水线（~3 分钟）
python run.py train --dry-run

# (b) 正式训练：2 万/科 × 1 epoch（RTX 3090 约 2.5 小时，显存峰值 ~18GB）
python run.py train

# 断点续训
python run.py train --resume-from outputs/lora_adapter/checkpoint-200
```

### 3. 合并 LoRA → HF 模型

```bash
python run.py merge
# 输出: .env 的 MERGED_MODEL_PATH（.env.example 模板为 ./finetune/base-model-finetuned；
#       未配置时为 ./outputs/merged_model），默认 fp16 ~16GB
```

### 4. 一键启动：OpenAI API + Web 聊天界面

```bash
python run.py serve
# 默认 0.0.0.0:8118（.env 的 API_HOST/API_PORT），API 与聊天网页共用同一端口
# 自动按 合并模型 → LoRA+基座 → 基座 的顺序加载微调模型，启动日志会打印实际来源
# 默认流式输出；采样默认值来自 .env 生成参数段；思考模式默认关闭
```

启动后两种使用方式：

- **Web 聊天界面**：浏览器打开 `http://<服务器IP>:8118`，直接对话（流式输出、多轮会话、停止/清空、示例问题）
- **OpenAI 兼容 API**：同一端口的 `/v1`，程序用 `openai` SDK 调用：

```python
from openai import OpenAI

client = OpenAI(base_url="http://<服务器IP>:8118/v1", api_key="empty")

# 默认即流式
stream = client.chat.completions.create(
    model="Qwen3-8B",                      # 服务端不校验；建议与 .env 的 SERVED_MODEL_NAME 一致
    messages=[{"role": "user", "content": "感冒发烧应该吃什么药？"}],
)
for chunk in stream:
    delta = chunk.choices[0].delta
    if delta and delta.content:
        print(delta.content, end="", flush=True)

# 非流式：传 stream=False；按请求覆盖采样参数；extra_body 平铺自定义字段
resp = client.chat.completions.create(
    model="Qwen3-8B",
    messages=[{"role": "user", "content": "儿童发烧多少度需要去医院？"}],
    stream=False,
    temperature=0.6,
    extra_body={"repetition_penalty": 1.05, "enable_thinking": False},
)
print(resp.choices[0].message.content)
```

---

## 致谢

- Unsloth 团队（训练加速）
- TRL / HuggingFace PEFT（SFT 框架）
- Qwen 团队（基座模型）
