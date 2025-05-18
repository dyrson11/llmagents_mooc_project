from pydantic import BaseModel
from typing import Optional, List

from langchain_core.messages import AnyMessage


# Define the input state
class AgentState(BaseModel):
    sentence: str
    source_language: str
    target_language: str
    dictionary_search: Optional[str] = None
    dictionary_analysis: Optional[str] = None
    translation: Optional[str] = None
    messages: Optional[List[AnyMessage]] = []
