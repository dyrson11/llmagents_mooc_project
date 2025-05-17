import os
import re
import json

from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace


def get_llm(
    model_name: str = "Qwen/Qwen3-30B-A3B", provider="fireworks-ai"
) -> ChatHuggingFace:
    """
    Initialize the model from Hugging Face.
    """
    llm = HuggingFaceEndpoint(
        # repo_id="Qwen/Qwen3-235B-A22B",
        repo_id=model_name,
        provider=provider,
        huggingfacehub_api_token=os.environ["HF_TOKEN"],
    )
    return llm


def try_parse_tool_calls(content: str):
    """Try parse the tool calls."""
    tool_calls = []
    offset = 0
    for i, m in enumerate(re.finditer(r"<tool_call>\n(.+)?\n</tool_call>", content)):
        if i == 0:
            offset = m.start()
        try:
            func = json.loads(m.group(1))
            tool_calls.append(
                {"type": "function", "function": func, "id": f"tool_call_id_{i+1}"}
            )
            if isinstance(func["arguments"], str):
                func["arguments"] = json.loads(func["arguments"])
        except json.JSONDecodeError as e:
            print(f"Failed to parse tool calls: the content is {m.group(1)} and {e}")
            pass
    if tool_calls:
        if offset > 0 and content[:offset].strip():
            c = content[:offset]
        else:
            c = ""
        return {"role": "assistant", "content": content, "tool_calls": tool_calls}
    return {"role": "assistant", "content": re.sub(r"<\|im_end\|>$", "", content)}
