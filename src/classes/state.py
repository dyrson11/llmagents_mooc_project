from typing import TypedDict, NotRequired, Required


# Define the input state
class AgentState(TypedDict, total=False):
    sentence: Required[str]
    source_language: Required[str]
    target_language: Required[str]
    dictionary_search: NotRequired[str]
    dictionary_analysis: NotRequired[str]
    translation: NotRequired[str]
