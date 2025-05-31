from dotenv import load_dotenv


from src.graph import Graph
from src.classes.state import AgentState
from src.utils.load_dataset import load_dataset

load_dotenv()


def main(
    dataset_path: str = "",
    percentage: int = 75,
    source_language: str = "es",
    target_language: str = "en",
):
    """
    Main function to run the agent with an initial state.
    """
    # Load the dataset
    dataset = load_dataset(dataset_path, percentage)

    # Create an instance of the Graph class
    graph = Graph(dataset=dataset, training=True)

    for index, row in dataset.iterrows():
        result = graph.run(
            input_state=AgentState(
                sentence=row["source_sentence"],
                source_language=source_language,
                target_language=target_language,
            )
        )
        if index > 20:
            break

    # Print the result
    # print("Final Result:")
    # print(result)


if __name__ == "__main__":
    dataset_path = "data/dataset/variantes_data/clean"
    percentage = 75
    source_language = "Aymara"
    target_language = "Spanish"
    main(dataset_path, percentage, source_language, target_language)
