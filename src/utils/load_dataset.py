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

    return (
        pd.DataFrame(data)
        .rename(columns={"es": "target_sentence", "aym": "source_sentence"})
        .reset_index(drop=True)
    )
