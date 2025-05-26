import re
import json
import pathlib
import logging
import pandas as pd

from langchain.tools import Tool
from typing import Dict, Any, cast
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

logger = logging.getLogger(__name__)

from ..utils.llm import get_llm
from ..classes.state import AgentState
from .prompt_rewriter import PromptRewriter
from ..config.prompts.evaluator import system_prompt, user_prompt


class TranslationEvaluator:
    """
    A class to interact with API to generate a dictionary of terms.
    """

    def __init__(self, dataset_dir) -> None:
        # Configure model
        self.llm = get_llm(model_name="Qwen/Qwen3-235B-A22B", provider="fireworks-ai")
        self.system_prompt = system_prompt
        self.user_prompt = user_prompt
        self.prompt_rewriter_agent = PromptRewriter(dataset_dir=dataset_dir)

        self.dataset = pd.read_csv(dataset_dir)

    def run(self, agent_state: AgentState) -> Dict[str, Any]:
        """
        Generate a dictionary of terms based on the research state.
        """
        source_sentence = agent_state.sentence
        target_sentence = self.dataset.loc[
            self.dataset["source_sentence"] == source_sentence, "target_sentence"
        ].values[0]

        system_message = SystemMessage(
            content=self.system_prompt.format(
                source_language=agent_state.source_language,
                target_language=agent_state.target_language,
            )
        )
        user_message = HumanMessage(
            content=self.user_prompt.format(
                sentence=source_sentence,
                reference=target_sentence,
                dictionary_assistant_analysis=agent_state.dictionary_analysis,
                translation_expert_analysis=agent_state.translation_analysis,
            )
        )
        response = self.llm.invoke([system_message, user_message])
        print(response)
        agent_state.messages.append(user_message)
        agent_state.messages.append(cast(AIMessage, response))
        print(response.content)

        # self.prompt_replacer(response.content, agent_state)
        self.existing_prompts = {
            "dictionary_assistant": agent_state.prompt_agent_dictionary_assistant,
            "translation_expert": agent_state.prompt_agent_translation_expert,
        }
        agent_state.changes = self.parse_handoffs(response.content, agent_state)
        return agent_state

    def parse_handoffs(self, response: str, state: AgentState) -> str:
        """
        Parse the response from the LLM to extract handoff information.
        """
        handoff_pattern = r"<problem_with_agent>(.*?)</problem_with_agent>"
        handoff_match = re.findall(handoff_pattern, response, re.DOTALL)
        changes = ""
        for handoff in handoff_match:
            handoff_dict = json.loads(handoff)
            agent_name = handoff_dict.get("agent_name")
            feedback = handoff_dict.get("feedback")
            if not agent_name:
                logger.warning("No agent name found in handoff.")
                continue
            if agent_name not in self.existing_prompts:
                logger.warning(f"Agent {agent_name} not found in existing prompts.")
                continue
            prompt = self.existing_prompts.get(agent_name)
            new_prompt, diff = self.prompt_rewriter_agent.run(prompt, feedback)
            state.__setattr__(f"prompt_agent_{agent_name}", new_prompt)

            self.write_prompt(agent_name, new_prompt)

            changes += f"Feedback for {agent_name}: {feedback}\n"
            changes += f"Changes made to its prompt: {diff}\n\n"

        return changes

    def write_prompt(self, agent_name: str, prompt) -> str:
        """
        Write the new prompt to the file.
        """
        for i in range(1, 10000):
            # check if the file exists with pathlib
            filename = f"output/new_prompts/{agent_name}/iteration_{i}.txt"
            if pathlib.Path(filename).exists():
                continue
            break

        with open(filename, "w") as f:
            f.write(prompt)
