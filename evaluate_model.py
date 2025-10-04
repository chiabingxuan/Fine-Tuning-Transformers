from eval_config import DATASET_PATH, FINETUNED_MODEL_NAME, FINETUNED_MODEL_REPO_ID, OUTPUTS_PATH
import json
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
import seaborn as sns
from transformers import AutoModelForSequenceClassification, Trainer, TrainingArguments
from utils import load_kaggle_dataset, tokenise_dataset


def make_and_save_predictions(trainer, test_dataset, folder_to_save):
    # Use trainer to predict on the test set
    predictions = trainer.predict(test_dataset)
    
    # Get predicted classes and actual classes
    logits = predictions.predictions
    y_preds = np.argmax(logits, axis=1)
    y_test = predictions.label_ids

    # Save predictions
    preds_df = pd.DataFrame(
        {
            "pred_class": y_preds,
            "actual_class": y_test
        }
    )
    preds_df.to_csv(os.path.join(folder_to_save, "preds.csv"), index=True)

    return y_preds, y_test


def display_and_save_cm(cm, folder_to_save) -> None:
    # Make heat map of confusion matrix
    plt.figure(figsize=(8, 6))
    sns.set_theme(font_scale=0.8)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")

    # Save this heat map
    cm_file_path = os.path.join(folder_to_save, "cm.png")
    plt.savefig(cm_file_path, bbox_inches="tight")
    print("Confusion matrix saved!")


def compute_and_save_eval_metrics(y_preds, y_test, folder_to_save):
    # Calculate evaluation metrics wrt fake news class label
    acc = accuracy_score(y_test, y_preds)
    precision = precision_score(y_test, y_preds, pos_label=1)
    recall = recall_score(y_test, y_preds, pos_label=1)
    f1 = f1_score(y_test, y_preds, pos_label=1)
    cm = confusion_matrix(y_test, y_preds)

    # Print evaluation metrics
    print(f"Accuracy: {round(acc * 100, 1)}%")
    print(f"Precision: {round(precision * 100, 1)}%")
    print(f"Recall: {round(recall * 100, 1)}%")
    print(f"F1 score: {round(f1 * 100, 1)}%")

    # Save metrics
    metrics = {
        "accuracy": acc,
        "precision": precision,
        "recall": recall,
        "f1": f1
    }
    with open(os.path.join(folder_to_save, "metrics.json"), "w") as json_file:
        json.dump(metrics, json_file, indent=4)

    # Display and save confusion matrix
    display_and_save_cm(cm=cm, folder_to_save=folder_to_save)


def evaluate_finetuned_model():
    # Load dataset
    dataset = load_kaggle_dataset(DATASET_PATH)

    # Tokenise dataset
    dataset, _ = tokenise_dataset(dataset=dataset, tokeniser_model_name=FINETUNED_MODEL_REPO_ID, text_column="text")

    # Load fine-tuned model
    model = AutoModelForSequenceClassification.from_pretrained(FINETUNED_MODEL_REPO_ID)

    # Default arguments for Trainer, just for the evaluation use-case
    test_args = TrainingArguments(
        do_train=False,
        do_predict=True,
        per_device_eval_batch_size=16,
        report_to="none"
    )

    # Wrap model in Trainer
    trainer = Trainer(
        model=model,
        args=test_args
    )

    # Get predictions on test set
    folder_to_save = os.path.normpath(os.path.join(OUTPUTS_PATH, FINETUNED_MODEL_NAME))
    os.makedirs(folder_to_save, exist_ok=True)
    y_preds, y_test = make_and_save_predictions(
        trainer=trainer,
        test_dataset=dataset["test"],
        folder_to_save=folder_to_save
    )

    # Get evaluation metrics
    compute_and_save_eval_metrics(y_preds=y_preds, y_test=y_test, folder_to_save=folder_to_save)