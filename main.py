from dotenv import load_dotenv
from evaluate_model import evaluate_finetuned_model
from finetune_model import run_finetuning_pipeline
from mode import IS_FINETUNING_MODE
import os

if __name__ == "__main__":
    load_dotenv()
    hf_token = os.getenv("HF_TOKEN")

    if IS_FINETUNING_MODE:
        # Finetune model based on training configurations
        run_finetuning_pipeline(hf_token=hf_token)
    
    else:
        # Evaluate based on chosen model in evaluation configurations
        evaluate_finetuned_model()