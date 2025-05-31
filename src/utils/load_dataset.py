import pathlib
import pandas as pd


def load_dataset(dataset_path: str, percentage: int) -> pd.DataFrame:
    """
    Load the dataset from the specified path.

    Args:
        dataset_path (str): The path to the dataset file.

    Returns:
        pd.DataFrame: The loaded dataset as a pandas DataFrame.
    """
    data = {}
    for lang in ["es", "aym"]:
        path = f"{dataset_path}_{percentage}.{lang}"
        with open(path, "r") as f:
            data[lang] = f.readlines()
            data[lang] = [line.strip() for line in data[lang]]
    dataset = (
        pd.DataFrame(data)
        .rename(columns={"es": "target_sentence", "aym": "source_sentence"})
        .reset_index(drop=True)
    )
    dataset = dataset[dataset["target_sentence"] != dataset["source_sentence"]]
    return dataset


def load_and_split_dataset(original_folder: str, cleaned_folder: str, data_path: str, dev_path: str):

    if pathlib.Path(f"{cleaned_folder}/train.csv").exists() and pathlib.Path(f"{cleaned_folder}/dev.csv").exists():
        train_clean = pd.read_csv(f"{cleaned_folder}/train.csv")
        train_original = pd.read_csv(f"{original_folder}/train.csv")
        dev_set = pd.read_csv(f"{cleaned_folder}/dev.csv")
        

    else:

        original_set = load_dataset(data_path, 0)
        cleaned_set = load_dataset(data_path, 100)

        dev_set = load_dataset(data_path, 100).sample(n=100, random_state=42)
        

        train_original = original_set[
            ~(original_set["source_sentence"].isin(dev_set["source_sentence"]))
        ].sample(n=500, random_state=42)
        train_clean = cleaned_set[
            ~(cleaned_set["source_sentence"].isin(dev_set["source_sentence"]))
        ].sample(n=500, random_state=42)

        train_clean.to_csv(f"{cleaned_folder}/train.csv", index=False)
        dev_set.to_csv(f"{cleaned_folder}/dev.csv", index=False)

        train_original.to_csv(f"{original_folder}/train.csv", index=False)
        dev_set.to_csv(f"{original_folder}/dev.csv", index=False)


    dataset_cleaned = {"train": train_clean, "dev": dev_set}
    dataset_original = {"train": train_original, "dev": dev_set}
    dataset = {"original": dataset_original, "cleaned": dataset_cleaned}

    return dataset
