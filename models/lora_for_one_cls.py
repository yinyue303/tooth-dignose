import json
import os
import shutil
import subprocess
from pathlib import Path
import argparse
from dataclasses import dataclass
import torch


# =========================
# 1. 跨平台路径配置
# =========================


def get_path_from_env(env_name: str, default: Path) -> Path:
    """优先读取环境变量，否则使用项目内的默认路径。"""

    configured_path = os.environ.get(env_name)

    if configured_path:
        return Path(configured_path).expanduser().resolve()

    return default.expanduser().resolve()


# 当前文件位于 <项目目录>/models/，因此默认取 models 的上一级作为项目目录。
# 将整个项目复制到 Linux 服务器后，只要目录结构不变，就不需要修改代码。
SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_PROJECT_DIR = SCRIPT_DIR.parent

PROJECT_DIR = get_path_from_env(
    "LORA_PROJECT_DIR",
    DEFAULT_PROJECT_DIR,
)
MODEL_DIR = get_path_from_env(
    "LORA_MODEL_DIR",
    PROJECT_DIR / "model" / "Qwen2.5-VL-7B-Instruct",
)
DATASET_DIR = get_path_from_env(
    "LORA_DATASET_DIR",
    PROJECT_DIR / "dataset",
)
TRAIN_FILE = DATASET_DIR / "train.json"
DATASET_INFO_FILE = DATASET_DIR / "dataset_info.json"
DATASET_NAME = "classification_sft"
OUTPUT_DIR = get_path_from_env(
    "LORA_OUTPUT_DIR",
    PROJECT_DIR / "saves" / "cls-SFT_7B_rank16_lr5e-4_epoch3",
)



# =========================
# 2. 数据集注册信息
# =========================

DATASET_INFO = {
    DATASET_NAME: {
        "file_name": "train.json",
        "formatting": "sharegpt",
        "columns": {
            "messages": "messages",
            "images": "images",
        },
        "tags": {
            "role_tag": "role",
            "content_tag": "content",
            "user_tag": "user",
            "assistant_tag": "assistant",
            "system_tag": "system",
        },
    }
}


def check_environment() -> None:
    """检查模型、数据、GPU 和训练程序。"""

    if not MODEL_DIR.is_dir():
        raise FileNotFoundError(f"模型目录不存在：{MODEL_DIR}")

    if not TRAIN_FILE.is_file():
        raise FileNotFoundError(f"训练数据不存在：{TRAIN_FILE}")

    if not torch.cuda.is_available():
        raise RuntimeError("PyTorch 没有检测到 CUDA。")

    gpu_count = torch.cuda.device_count()


    print(f"检测到 {gpu_count} 张 GPU：")

    for index in range(gpu_count):
        print(f"  GPU {index}: {torch.cuda.get_device_name(index)}")

    if shutil.which("llamafactory-cli") is None:
        raise RuntimeError(
            "没有找到 llamafactory-cli，请先安装 LLaMA-Factory。"
        )


def create_dataset_info() -> None:
    """创建 LLaMA-Factory 所需的数据集注册文件。"""

    DATASET_DIR.mkdir(parents=True, exist_ok=True)

    with DATASET_INFO_FILE.open("w", encoding="utf-8") as file:
        json.dump(
            DATASET_INFO,
            file,
            ensure_ascii=False,
            indent=2,
        )

    print(f"已生成数据集注册文件：{DATASET_INFO_FILE}")


def validate_dataset() -> None:
    """验证图片路径和 <image> 占位符数量。"""

    print("正在检查训练数据……")

    with TRAIN_FILE.open("r", encoding="utf-8") as file:
        records = json.load(file)

    errors = []

    for index, record in enumerate(records):
        messages = record.get("messages", [])
        images = record.get("images", [])

        image_token_count = sum(
            message.get("content", "").count("<image>")
            for message in messages
        )

        if image_token_count != len(images):
            errors.append(
                f"第 {index} 条："
                f"<image> 数量={image_token_count}，"
                f"images 数量={len(images)}"
            )

        for relative_path in images:
            image_path = DATASET_DIR / relative_path

            if not image_path.is_file():
                errors.append(
                    f"第 {index} 条图片不存在：{image_path}"
                )

        if len(errors) >= 100:
            break

    print(f"训练样本数量：{len(records)}")

    if errors:
        print("发现以下数据问题：")

        for error in errors:
            print(f"  {error}")

        raise RuntimeError(
            "数据检查失败，请修复数据后重新运行。"
        )

    print("数据检查通过。")


def build_training_command() -> list[str]:
    """生成 LLaMA-Factory 训练命令。"""

    return [
        "llamafactory-cli",
        "train",

        # 模型
        "--model_name_or_path",
        str(MODEL_DIR),
        "--trust_remote_code",
        "true",
        "--freeze_vision_tower",
        "true",
        "--freeze_multi_modal_projector",
        "true",

        # SFT 与 LoRA
        "--stage",
        "sft",
        "--do_train",
        "true",
        "--finetuning_type",
        "lora",
        "--lora_target",
        "all",
        "--lora_rank",
        "16",
        "--lora_alpha",
        "32",
        "--lora_dropout",
        "0.05",

        # 数据集
        "--dataset_dir",
        str(DATASET_DIR),
        "--dataset",
        DATASET_NAME,
        "--template",
        "qwen2_vl",
        "--cutoff_len",
        "4096",
        "--max_samples",
        "30000",
        "--overwrite_cache",
        "true",
        "--preprocessing_num_workers",
        "16",
        "--dataloader_num_workers",
        "4",

        # 输出
        "--output_dir",
        str(OUTPUT_DIR),
        "--logging_steps",
        "10",
        "--save_steps",
        "50",
        "--save_total_limit",
        "3",
        "--plot_loss",
        "true",
        "--overwrite_output_dir",
        "true",
        "--report_to",
        "none",

        # 训练参数
        "--per_device_train_batch_size",
        "1",
        "--gradient_accumulation_steps",
        "8",
        "--learning_rate",
        "0.0005",
        "--num_train_epochs",
        "3",
        "--lr_scheduler_type",
        "cosine",
        "--warmup_ratio",
        "0.1",
        "--weight_decay",
        "0.0",
        "--max_grad_norm",
        "1.0",
        "--bf16",
        "true",
        "--tf32",
        "true",
        "--ddp_timeout",
        "180000000",
    ]


def train() -> None:
  
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    environment = os.environ.copy()

    environment.setdefault(
    "CUDA_VISIBLE_DEVICES",
    "0,1,2,3,4",
)




    command = build_training_command()

    print("\n准备启动训练：")
    print(" ".join(f'"{item}"' if " " in item else item for item in command))
    print()

    subprocess.run(
        command,
        env=environment,
        check=True,
    )


def main() -> None:
    check_environment()
    create_dataset_info()
    validate_dataset()
    train()


if __name__ == "__main__":
    main()
