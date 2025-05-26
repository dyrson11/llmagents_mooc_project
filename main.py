from src.graph import Graph
from dotenv import load_dotenv

load_dotenv()


def main(
    sentence: str = "Hello, how are you?",
    source_language: str = "es",
    target_language: str = "en",
):
    """
    Main function to run the agent with an initial state.
    """
    # Create an instance of the Graph class
    graph = Graph(
        sentence, source_language, target_language, dataset_path="data/dataset/test.csv"
    )

    result = graph.run()

    # Print the result
    print("Final Result:")
    print(result)


if __name__ == "__main__":
    sentence = "Wali askinakw yatiqawayta, ukhamarak yaqha jaqinakarus uñt 'awayta."
    source_language = "Aymara"
    target_language = "Spanish"
    main(sentence, source_language, target_language)
