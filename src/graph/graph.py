from langgraph.graph import StateGraph
from langgraph.prebuilt import ToolNode, tools_condition

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
        sentence: str = "",
        source_language: str = "es",
        target_language: str = "en",
        dataset_path: str = "",
    ):
        self.input_state = AgentState(
            sentence=sentence,
            source_language=source_language,
            target_language=target_language,
        )
        self.dataset_path = dataset_path
        self._init_nodes()
        self.create_graph()

    def _init_nodes(self):
        self.dictionary_agent = DictionaryAssistant()
        self.translation_agent = TranslationExpert()
        self.llm_evaluator = TranslationEvaluator(dataset_dir=self.dataset_path)

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
        self.graph.add_edge("translation_expert", "llm_evaluator")
        self.graph.set_entry_point("start")
        self.graph.set_finish_point("llm_evaluator")

    def run(self):
        """
        Run the agent with the input state.
        """
        compiled_graph = self.graph.compile()
        # Run the agent with the input state
        result = compiled_graph.invoke(self.input_state)
        return result
