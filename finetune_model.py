import evaluate
from finetune_config import SEED, DATASET_PATH, MODEL_NAME, FINETUNED_MODEL_NAME, FINETUNED_MODEL_REPO_ID, MODEL_WEIGHTS_PATH, NUM_CLASSES, TRACKIO_SPACE_ID, TRACKIO_PROJECT, LORA_ENABLED, QUANTISATION_ENABLED, train_config, quantisation_config, lora_config
from huggingface_hub import login
import numpy as np
import os
from peft import get_peft_model
import random
import torch
from transformers import AutoModelForSequenceClassification, set_seed, Trainer
from utils import load_kaggle_dataset, tokenise_dataset


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


def finetune_model(model, model_name, tokeniser, finetuned_model_name, repo_id, dataset, model_weights_path):
    os.makedirs(model_weights_path, exist_ok=True)
    metric = evaluate.load("accuracy")

    print(f"Fine-tuning {model_name}...")
    trainer = Trainer(
        model=model,
        args=train_config,
        train_dataset=dataset["train"],
        eval_dataset=dataset["val"],
        compute_metrics=lambda eval_pred: compute_metrics(metric, eval_pred),
    )
    trainer.train()
    print("Fine-tuning completed!")

    # Add model and tokeniser to Hugging Face hub
    trainer.push_to_hub()
    tokeniser.push_to_hub(repo_id)
    print(f"Fine-tuned model {finetuned_model_name} and tokeniser uploaded to Hugging Face Hub!")

    return model


def run_finetuning_pipeline(hf_token):
    # Log in to Hugging Face
    login(hf_token)

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
    dataset, tokeniser = tokenise_dataset(dataset=dataset, tokeniser_model_name=MODEL_NAME, text_column="text")

    # Fine-tune model
    model = load_model(
        model_name=MODEL_NAME,
        num_classes=NUM_CLASSES,
        quantisation_enabled=QUANTISATION_ENABLED,
        lora_enabled=LORA_ENABLED
    )
    model_ft = finetune_model(
        model=model,
        model_name=MODEL_NAME,
        tokeniser=tokeniser,
        finetuned_model_name=FINETUNED_MODEL_NAME,
        repo_id=FINETUNED_MODEL_REPO_ID,
        dataset=dataset,
        model_weights_path=MODEL_WEIGHTS_PATH
    )