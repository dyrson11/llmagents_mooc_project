import pandas as pd
from langgraph.graph import StateGraph
from langgraph.prebuilt import ToolNode, tools_condition

from ..utils.metrics import calculate_score_report
from ..classes.state import AgentState
from ..agents.dictionary_assistant import DictionaryAssistant
from ..agents.translation_expert import TranslationExpert
from ..agents.translation_evaluator import TranslationEvaluator
from ..agents.prompt_rewriter import PromptRewriter


class Graph:
    """
    This class defines the state graph for the agent.
    """

    def __init__(
        self,
        dataset: pd.DataFrame = pd.DataFrame(),
        training: bool = False,
    ):
        self.dataset = dataset
        self.training = training
        self._init_nodes()
        self.create_graph()

    def _init_nodes(self):
        self.dictionary_agent = DictionaryAssistant()
        self.translation_agent = TranslationExpert()
        self.llm_evaluator = TranslationEvaluator(dataset=self.dataset)

    def get_graph(self):
        return self.graph

    def start(self, state: AgentState):
        """Initialize the workflow with the input message."""
        return state

    def create_graph(self):
        """
        Create the state graph for the agent.
        """

        self.graph = StateGraph(AgentState)
        self.graph.add_node("start", self.start)

        # Define nodes: these do the work
        self.graph.add_node("dictionary_assistant", self.dictionary_agent.run)
        # self.graph.add_node("tools", ToolNode(tools))
        self.graph.add_node("translation_expert", self.translation_agent.run)
        self.graph.add_node("llm_evaluator", self.llm_evaluator.run)

        # Define edges: these determine how the control flow moves
        self.graph.add_edge("start", "dictionary_assistant")
        # self.graph.add_conditional_edges(
        #     "dictionary_assistant",
        #     # If the latest message requires a tool, route to tools
        #     # Otherwise, provide a direct response
        #     tools_condition,
        # )
        # self.graph.add_edge("tools", "dictionary_assistant")

        # self.graph.add_conditional_edges(
        #     "dictionary_assistant",
        #     # If the latest message requires a tool, route to tools
        #     # Otherwise, provide a direct response
        #     tools_condition,
        # )
        self.graph.add_edge("dictionary_assistant", "translation_expert")
        if self.training:
            self.graph.add_edge("translation_expert", "llm_evaluator")
            self.graph.set_finish_point("llm_evaluator")
        else:
            self.graph.set_finish_point("translation_expert")
        self.graph.set_entry_point("start")

    def run(self, input_state: AgentState) -> AgentState:
        """
        Run the agent with the input state.
        """
        compiled_graph = self.graph.compile()
        # Run the agent with the input state
        result = compiled_graph.invoke(input_state)
        final_translation = result["translation"]
        print("")
        print("=" * 50)
        print(f"Source Sentence: {result['sentence']}")
        print(f"Final Translation: {final_translation}")
        if self.training:
            print(f"Target Sentence: {result['target_sentence']}")
            if len(final_translation.strip()) == 0:
                bleu = 0
                chrf = 0
            else:
                bleu, chrf = calculate_score_report(
                    final_translation, [result["target_sentence"]], score_only=True
                )
            print(f"BLEU: {bleu}, CHRF: {chrf}")
            print("=" * 50)
            return final_translation, bleu, chrf, result["messages"]
        else:
            print("=" * 50)
            return final_translation, None, None, result["messages"]
