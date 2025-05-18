import logging

from typing import Dict, Any, cast
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

logger = logging.getLogger(__name__)

from ..classes.state import AgentState
from ..utils.llm import get_llm
from ..config.prompts.translation import system_prompt, user_prompt


class TranslationExpert:
    """
    A class to interact with API to generate a dictionary of terms.
    """

    def __init__(self) -> None:
        # Configure model
        self.llm = get_llm(model_name="Qwen/Qwen3-30B-A3B", provider="fireworks-ai")
        self.system_prompt = system_prompt
        self.user_prompt = user_prompt

    def run(self, agent_state: AgentState) -> Dict[str, Any]:
        """
        Generate a dictionary of terms based on the research state.
        """

        system_message = SystemMessage(
            content=self.system_prompt.format(
                SOURCE_LANGUAGE=agent_state.source_language,
                TARGET_LANGUAGE=agent_state.target_language,
            )
        )
        user_message = HumanMessage(
            content=self.user_prompt.format(
                sentence=agent_state.sentence,
                dictionary_assistant_analysis=agent_state.dictionary_analysis,
            )
        )
        agent_state.messages.append(user_message)

        response = self.llm.invoke([system_message, user_message])
        agent_state.messages.append(cast(AIMessage, response))
        print(response.content)

        # # Get the model's response
        # response = cast(AIMessage, parsed_response)
        return agent_state
