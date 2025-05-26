import re
import pathlib
import logging
import pandas as pd

from langchain.tools import Tool
from typing import Dict, Any
from langchain_core.messages import HumanMessage, SystemMessage

logger = logging.getLogger(__name__)

from ..utils.llm import get_llm
from ..config.prompts.rewriter import system_prompt, user_prompt


class PromptRewriter:
    """
    A class to interact with API to generate a dictionary of terms.
    """

    def __init__(self, dataset_dir) -> None:
        # Configure model
        self.llm = get_llm(model_name="Qwen/Qwen3-235B-A22B", provider="fireworks-ai")
        self.tools = self._init_tools()
        self.llm = self.llm.bind_tools(self.tools)
        self.system_prompt = system_prompt
        self.user_prompt = user_prompt

        self.dataset = pd.read_csv(dataset_dir)

    def _init_tools(self):
        """
        Initialize the tools for the LLM.
        """
        replace_prompt_tool = Tool(
            name="replace_prompt",
            func=self.replace_tool,
            description="""Replace the text in the prompt with the new one.
        Parameters:
            text_to_replace (str): The text to replace in the prompt.
            new_text (str): The new text to replace with.""",
        )
        return [replace_prompt_tool]

    def run(self, prompt: str, feedback: str) -> Dict[str, Any]:
        """
        Edit a prompt based on the feedback.
        Parameters:
            prompt (str): The prompt to edit.
            feedback (str): The feedback to apply to the prompt.
        """

        system_message = SystemMessage(content=self.system_prompt)
        user_message = HumanMessage(
            content=self.user_prompt.format(
                prompt=prompt,
                feedback=feedback,
            )
        )
        new_prompt = prompt
        changes = None
        for _ in range(5):
            response = self.llm.invoke([system_message, user_message])
            print(response.content)
            # check if response has tool calls
            if not hasattr(response, "tool_calls") or not response.tool_calls:
                continue
            print(response.tool_calls)
            for tool_call in response.tool_calls:
                if "name" not in tool_call:
                    print("Tool call does not have a name attribute. Skipping.")
                    continue
                if tool_call["name"] != "replace_prompt":
                    print(
                        f"Tool call name '{tool_call['name']}' does not match 'replace_prompt'. Skipping."
                    )
                    continue
                # Use the tool to replace the prompt
                new_prompt, changes = self.replace_tool(
                    new_prompt,
                    tool_call["args"].get("text_to_replace"),
                    tool_call["args"].get("new_text"),
                )
            break
        if new_prompt is None or changes is None:
            logger.warning(
                "Tool call failed or did not return the expected format. "
                "Returning the original prompt without changes."
            )
            new_prompt = prompt
            changes = "No changes made to the prompt due to tool call failure."
        return new_prompt, changes

    def replace_tool(self, prompt: str, text_to_replace: str, new_text: str):
        """
        Replace the text in the prompt with the new one.
        Parameters:
            text_to_replace (str): The text to replace in the prompt.
            new_text (str): The new text to replace with.
        """
        print(f"Replacing '{text_to_replace}' with '{new_text}' in the prompt.")
        if text_to_replace not in prompt:
            print(
                f"Prompt for does not contain the string to replace. Please check the prompt."
            )
            return "", "No changes made to the prompt."

        new_prompt = prompt.replace(text_to_replace, new_text)
        changes = (
            f"<previous_prompt_fragment>{text_to_replace}</previous_prompt_fragment>"
        )
        changes += f"<new_prompt_fragment>{new_text}</new_prompt_fragment>"

        return new_prompt, changes
