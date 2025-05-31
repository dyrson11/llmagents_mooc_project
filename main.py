import os
import pandas as pd
from tqdm import tqdm
from dotenv import load_dotenv

from src.graph import Graph
from src.classes.state import AgentState
from src.utils.load_dataset import load_and_split_dataset

load_dotenv()


def cycle(
    dataset: pd.DataFrame,
    source_language: str,
    target_language: str,
    output_file: str,
    training: str,
    starts_from_idx: int = 0,
):
    graph = Graph(dataset=dataset, training=training)
    result_list = []
    bleu_list = []
    chrf_list = []
    messages = []
    folder = "/".join(output_file.split("/")[:-1])
    os.makedirs(folder, exist_ok=True)
    if os.path.exists(output_file):
        with open(output_file, "r") as f:
            result_list = f.read().splitlines()
        with open(f"{folder}/bleu.txt", "r") as f:
            bleu_list = f.read().splitlines()
        with open(f"{folder}/chrf.txt", "r") as f:
            chrf_list = f.read().splitlines()

    messages_folder = f"{"/".join(output_file.split("/")[:-2])}/messages"
    os.makedirs(messages_folder, exist_ok=True)
    for index, row in tqdm(dataset.iterrows(), total=dataset.shape[0]):
        if index < starts_from_idx:
            continue
        result, bleu, chrf, messages = graph.run(
            input_state=AgentState(
                sentence=row["source_sentence"],
                source_language=source_language,
                target_language=target_language,
            )
        )
        result_list.append(result.strip() if result is not None else "")
        bleu_list.append(str(bleu))
        chrf_list.append(str(chrf))
        with open(output_file, "w") as f:
            f.write("\n".join(result_list))
        with open(f"{folder}/bleu.txt", "w") as f:
            f.write("\n".join(bleu_list))
        with open(f"{folder}/chrf.txt", "w") as f:
            f.write("\n".join(chrf_list))
        with open(f"{messages_folder}/{index}.txt", "w") as f:
            f.write("\n".join(message.__str__() for message in messages))


def main(
    dataset_path: str = "",
    original_folder: str = "",
    cleaned_folder: str = "",
    source_language: str = "es",
    target_language: str = "en",
    mode: str = "dev",
    dtype: str = "original",
    starts_from_idx: int = 0,

    dev_path: str = "",
):
    """
    Main function to run the agent with an initial state.
    """
    # Load the dataset
    dataset = load_and_split_dataset(original_folder, cleaned_folder, dataset_path, dev_path)
    training = True if mode == "train" else False
    cycle(
        dataset[dtype][mode],
        source_language,
        target_language,
        f"output/translations/{dtype}/{mode}/output.txt",
        training=training,
        starts_from_idx=starts_from_idx,
    )
    # Create an instance of the Graph class

    # Print the result
    # print("Final Result:")
    # print(result)


if __name__ == "__main__":
    dataset_path = "data/dataset/variantes_data/clean"
    original_folder = "data/dataset/original"
    cleaned_folder = "data/dataset/cleaned"
    source_language = "Aymara"
    target_language = "Spanish"
    mode = "dev"
    dtype = "original"
    starts_from_idx = 0
    main(
        dataset_path,
        original_folder,
        cleaned_folder,
        source_language,
        target_language,
        mode,
        dtype,
        starts_from_idx=starts_from_idx,
    )
