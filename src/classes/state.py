import pandas as pd
from pydantic import BaseModel
from typing import Optional, List

from langchain_core.messages import AnyMessage


# Define the input state
class AgentState(BaseModel):
    sentence: str
    source_language: str
    target_language: str
    target_sentence: Optional[str] = None
    dictionary_search: Optional[str] = None
    dictionary_analysis: Optional[str] = None
    translation_analysis: Optional[str] = None
    translation: Optional[str] = None
    messages: Optional[List[AnyMessage]] = []
    prompt_agent_translation_expert: Optional[str] = None
    prompt_agent_dictionary_assistant: Optional[str] = None
    changes: Optional[str] = None
