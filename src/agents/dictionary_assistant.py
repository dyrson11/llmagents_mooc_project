import re
import copy
import json
import pathlib
import logging

from typing import Dict, Any, cast
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

logger = logging.getLogger(__name__)

from ..classes.state import AgentState
from ..utils.dictionary_search import DictionarySearch
from ..classes.pydantic_model import Palabra, Sufijo, Posposicion
from ..utils.llm import get_llm, try_parse_tool_calls, invoke_llm
from ..config.prompts.dictionary import system_prompt, user_prompt


class DictionaryAssistant:
    """
    A class to interact with API to generate a dictionary of terms.
    """

    def __init__(
        self,
        system_prompt_path="output/new_prompts/dictionary_assistant/iteration_{i}.txt",
    ) -> None:
        # Configure model
        self.llm = get_llm(model_name="Qwen/Qwen3-30B-A3B", provider="fireworks-ai")
        self.system_prompt_path = system_prompt_path
        self.system_prompt = system_prompt
        self.user_prompt = user_prompt
        self.dictionary, self.suffixes = self.load_dictionary(
            "data/diccionarios/aymara/dic-2-ay-es.json"
        )

        self.dictionary_search = DictionarySearch(self.dictionary)

        # Initialize context dictionary for use across methods
        self.context = {
            "company": "Unknown Company",
            "industry": "Unknown",
            "hq_location": "Unknown",
        }

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

    def load_dictionary(self, dictionary_path: str) -> Dict[str, Any]:
        """
        Load the dictionary from a file.
        """
        with open(dictionary_path, "r", encoding="utf-8") as f:
            dictionary = json.load(f)
            dictionary = {
                "diccionario": dictionary["diccionario"],
                "version": dictionary["version"],
                "fecha": dictionary["fecha"],
                "fuente": dictionary["fuente"],
                "palabras": {
                    key: Palabra(**value)
                    for key, value in dictionary["palabras"].items()
                },
                "sufijos": {
                    key: Sufijo(**value) for key, value in dictionary["sufijos"].items()
                },
                "posposiciones": {
                    key: Posposicion(**value)
                    for key, value in dictionary["posposiciones"].items()
                },
            }

        palabras_formateadas = {}
        for key, value in dictionary["palabras"].items():
            key = key.replace("'", "’")
            palabras_formateadas[key] = copy.deepcopy(value)
            palabras_formateadas[key].palabra = key

        palabras_sin_sufijo = {}
        counter = 0
        for key, value in palabras_formateadas.items():
            tiene_sufijo = False
            for sufijo in dictionary["sufijos"].values():
                if key.endswith(sufijo.sufijo):
                    if len(key) == len(sufijo.sufijo):
                        continue
                    palabras_sin_sufijo[key[: -len(sufijo.sufijo)]] = value
                    tiene_sufijo = True
                    counter += 1
                    break
            if not tiene_sufijo:
                palabras_sin_sufijo[key] = value

        diccionario_general = copy.deepcopy(palabras_sin_sufijo)
        diccionario_general.update(palabras_formateadas)
        diccionario_general.update(dictionary["posposiciones"])
        return diccionario_general, dictionary["sufijos"]

    def run(self, agent_state: AgentState) -> Dict[str, Any]:
        """
        Generate a dictionary of terms based on the research state.
        """

        dictionary_result, sentence = self.dictionary_search.process_sentence(
            agent_state.sentence,
            self.dictionary,
            self.suffixes,
        )
        agent_state.dictionary_search = dictionary_result
        self.load_system_prompt(self.system_prompt_path)
        agent_state.prompt_agent_dictionary_assistant = self.system_prompt

        # system_prompt = self.system_prompt.format(
        #     SOURCE_LANGUAGE=agent_state.source_language,
        #     TARGET_LANGUAGE=agent_state.target_language,
        # )
        system_prompt = self.system_prompt.replace(
            "{SOURCE_LANGUAGE}", agent_state.source_language
        ).replace(
            "{TARGET_LANGUAGE}", agent_state.target_language
        )

        system_message = SystemMessage(content=system_prompt)
        user_message = HumanMessage(
            content=self.user_prompt.format(
                SENTENCE=agent_state.sentence,
                DICTIONARY=dictionary_result,
            )
        )

        agent_state.messages.append(system_message)
        agent_state.messages.append(user_message)

        # response = self.llm.invoke([system_message, user_message])
        response = invoke_llm(self.llm, system_message, user_message)

        # print(response.content)

        dictionary_analysis = re.search(
            r"<dictionary_assistant_analysis>(.*?)</dictionary_assistant_analysis>",
            response.content,
            re.DOTALL,
        )

        agent_state.dictionary_analysis = (
            dictionary_analysis.group(1) if dictionary_analysis else None
        )

        if response.additional_kwargs.get("tool_calls"):
            return agent_state

        # Parse the response to extract the tool calls
        parsed_response = try_parse_tool_calls(response.content)
        agent_state.messages.append(cast(AIMessage, response))

        # # Get the model's response
        # response = cast(AIMessage, parsed_response)
        return agent_state
