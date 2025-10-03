from datasets import DatasetDict
from kagglehub import KaggleDatasetAdapter, dataset_load
from transformers import AutoTokenizer


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
    print(f"Dataset loaded from {dataset_path} on Kaggle!\n")

    return dataset


def tokenise(tokeniser, example, text_column):
    return tokeniser(example[text_column], padding="max_length", truncation=True)


def tokenise_dataset(dataset, tokeniser_model_name, text_column):
    tokeniser = AutoTokenizer.from_pretrained(tokeniser_model_name)
    dataset = dataset.map(lambda example: tokenise(tokeniser, example, text_column), batched=True)
    print(f"Tokenisation completed with {tokeniser_model_name}!\n")

    return dataset, tokeniser