import os
from peft import LoraConfig, TaskType
import torch
from transformers import BitsAndBytesConfig, TrainingArguments

# General configurations
SEED = 42
DATASET_PATH = "bingxuanchia/cleaned-isot-fake-news-dataset"
MODEL_NAME = "google-bert/bert-base-cased"
FINETUNED_MODEL_NAME = "isot-bert-finetuned"
MODEL_WEIGHTS_PATH = "weights"   # Path which will store the saved weights
NUM_CLASSES = 2                  # Number of distinct class labels
TRACKIO_SPACE_ID = "chiabingxuan/ISOTFineTuning"
TRACKIO_PROJECT = "bert-finetuning"

# Training configurations
train_config = TrainingArguments(
    # Model and data
    output_dir=os.path.join(MODEL_WEIGHTS_PATH, FINETUNED_MODEL_NAME),

    # Training hyperparameters
    learning_rate=5e-5,
    num_train_epochs=5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    eval_strategy="steps",
    seed=SEED,

    # Hugging Face Hub integration
    push_to_hub=True,

    # Experiment tracking
    logging_steps=250,
    report_to=["trackio"],
    run_name=FINETUNED_MODEL_NAME
)

# Quantisation configurations
QUANTISATION_ENABLED = False
quantisation_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

# LoRA configurations
LORA_ENABLED = False
lora_config = LoraConfig(
    r=16,
    lora_alpha=16,
    lora_dropout=0.1,
    bias="none",
    task_type=TaskType.SEQ_CLS
)