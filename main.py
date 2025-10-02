from config import SEED, DATASET_PATH, MODEL_NAME, FINETUNED_MODEL_NAME, MODEL_WEIGHTS_PATH, NUM_CLASSES, TRACKIO_SPACE_ID, TRACKIO_PROJECT, LORA_ENABLED, QUANTISATION_ENABLED, train_config, quantisation_config, lora_config
from datasets import DatasetDict
from dotenv import load_dotenv
import evaluate
from huggingface_hub import login
from kagglehub import KaggleDatasetAdapter, dataset_load
import numpy as np
import os
from peft import get_peft_model
import random
import time
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer, set_seed, Trainer


def set_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)           # CPU
    torch.cuda.manual_seed(seed)      # Current GPU
    torch.cuda.manual_seed_all(seed)  # All GPUs (if multiple)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    set_seed(SEED)


def check_device():
    if torch.cuda.is_available():
        device = "cuda"
        print(f"Using CUDA GPU: {torch.cuda.get_device_name()}")
        print(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}GB\n")

    else:
        device = "cpu"
        print("Using CPU\n")
    
    return device


def load_kaggle_dataset(dataset_path):
    dataset = dict()
    for phase in ["train", "val", "test"]:
        phase_data = dataset_load(
            KaggleDatasetAdapter.HUGGING_FACE,
            dataset_path,
            f"{phase}.csv"
        )
        
        # Rename the "class" column to "labels" column for BERT
        phase_data = phase_data.rename_column("class", "labels")

        dataset[phase] = phase_data
    dataset = DatasetDict(dataset)
    print(f"Dataset loaded from {DATASET_PATH} on Kaggle!\n")

    return dataset


def tokenise(tokeniser, example, text_column):
    return tokeniser(example[text_column], padding="max_length", truncation=True)


def tokenise_dataset(dataset, tokeniser_model_name, text_column):
    tokeniser = AutoTokenizer.from_pretrained(tokeniser_model_name)
    dataset = dataset.map(lambda example: tokenise(tokeniser, example, text_column), batched=True)
    print(f"Tokenisation completed with {tokeniser_model_name}!\n")

    return dataset


def compute_metrics(eval_metric, eval_pred):
    logits, labels = eval_pred

    # Convert the logits to their corresponding predictions
    predictions = np.argmax(logits, axis=-1)
    return eval_metric.compute(predictions=predictions, references=labels)


def load_model(model_name, num_classes, quantisation_enabled, lora_enabled):
    if quantisation_enabled:
        print("Quantisation enabled")
        model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            num_labels=num_classes,
            device_map="auto",
            quantization_config=quantisation_config
        )

    else:
        model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            num_labels=num_classes,
            device_map="auto"
        )

    if lora_enabled:
        model = get_peft_model(model, lora_config)
        print(f"LoRA enabled:")
        model.print_trainable_parameters()
        print()

    print(f"Model architecture:")
    print(model)
    print()

    return model


def finetune(model, model_name, finetuned_model_name, dataset, model_weights_path):
    os.makedirs(model_weights_path, exist_ok=True)
    metric = evaluate.load("accuracy")

    print(f"Fine-tuning {model_name}...")
    start = time.time()
    trainer = Trainer(
        model=model,
        args=train_config,
        train_dataset=dataset["train"],
        eval_dataset=dataset["val"],
        compute_metrics=lambda eval_pred: compute_metrics(metric, eval_pred),
    )
    trainer.train()
    end = time.time()
    time_elapsed = int(end - start)
    print("Fine-tuning completed in {:.0f}m {:.0f}s".format(time_elapsed // 60, time_elapsed % 60))

    # Add model and tokeniser to Hugging Face hub
    trainer.push_to_hub()
    print(f"Fine-tuned model {finetuned_model_name} and tokeniser uploaded to Hugging Face Hub!")

    return model


if __name__ == "__main__":
    # Log in to Hugging Face
    load_dotenv()
    login(os.getenv("HF_TOKEN"))

    # To conduct experiment tracking with trackio
    # Dashboard viewable on Hugging Face Spaces
    os.environ["TRACKIO_SPACE_ID"] = TRACKIO_SPACE_ID
    os.environ["TRACKIO_PROJECT"] = TRACKIO_PROJECT

    # Set seed for reproducibility
    set_seeds(SEED)
    print(f"Seed: {SEED}\n")

    # Check device
    device = check_device()

    # Load and verify dataset
    dataset = load_kaggle_dataset(dataset_path=DATASET_PATH)
    
    # Tokenise dataset
    dataset = tokenise_dataset(dataset=dataset, tokeniser_model_name=MODEL_NAME, text_column="text")

    # Fine-tune model
    model = load_model(
        model_name=MODEL_NAME,
        num_classes=NUM_CLASSES,
        quantisation_enabled=QUANTISATION_ENABLED,
        lora_enabled=LORA_ENABLED
    )
    model_ft = finetune(
        model=model,
        model_name=MODEL_NAME,
        finetuned_model_name=FINETUNED_MODEL_NAME,
        dataset=dataset,
        model_weights_path=MODEL_WEIGHTS_PATH
    )