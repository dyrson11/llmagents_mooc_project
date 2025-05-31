import re
import pathlib
import logging

from typing import Dict, Any, cast
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

logger = logging.getLogger(__name__)

from ..classes.state import AgentState
from ..utils.llm import get_llm, invoke_llm
from ..config.prompts.translation import system_prompt, user_prompt


class TranslationExpert:
    """
    A class to interact with API to generate a dictionary of terms.
    """

    def __init__(
        self,
        system_prompt_path="output/new_prompts/translation_expert/iteration_{i}.txt",
    ) -> None:
        # Configure model
        self.llm = get_llm(model_name="Qwen/Qwen3-30B-A3B", provider="fireworks-ai")
        self.system_prompt_path = system_prompt_path
        self.user_prompt = user_prompt

    def load_system_prompt(self, path) -> str:
        """
        Load the system prompt from a file.
        """
        for i in range(1, 10000):
            # check if the file exists with pathlib
            if pathlib.Path(path.format(i=i)).exists():
                continue
            if i == 1:
                # If the file does not exist, use the default system prompt
                self.system_prompt = system_prompt
                break
            with open(path.format(i=i - 1), "r", encoding="utf-8") as f:
                self.system_prompt = f.read()
            break

    def run(self, agent_state: AgentState) -> Dict[str, Any]:
        """
        Generate a dictionary of terms based on the research state.
        """

        self.load_system_prompt(self.system_prompt_path)

        system_prompt = self.system_prompt.format(
            SOURCE_LANGUAGE=agent_state.source_language,
            TARGET_LANGUAGE=agent_state.target_language,
        )
        agent_state.prompt_agent_translation_expert = system_prompt

        system_message = SystemMessage(content=system_prompt)
        user_message = HumanMessage(
            content=self.user_prompt.format(
                sentence=agent_state.sentence,
                dictionary_assistant_analysis=agent_state.dictionary_analysis,
            )
        )
        agent_state.messages.append(system_message)
        agent_state.messages.append(user_message)

        response = invoke_llm(self.llm, system_message, user_message)
        agent_state.messages.append(cast(AIMessage, response))
        agent_state.translation_analysis = response.content
        translation = re.search(
            r"<final_translation>\s*(.*?)\s*</final_translation>",
            response.content,
            re.IGNORECASE + re.DOTALL,
        )
        agent_state.translation = translation.group(1) if translation else ""
        print(response.content)

        # # Get the model's response
        # response = cast(AIMessage, parsed_response)
        return agent_state
