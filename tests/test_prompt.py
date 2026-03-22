import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from inference.prompt import build_system_prompt, build_user_prompt, build_chat_messages


def test_system_prompt_includes_business_name():
    prompt = build_system_prompt("Sweet Rise Bakery")
    assert "Sweet Rise Bakery" in prompt
    assert "ONLY" in prompt


def test_user_prompt_includes_context_and_query():
    prompt = build_user_prompt("We have vegan cakes.", "Do you have vegan options?")
    assert "We have vegan cakes." in prompt
    assert "Do you have vegan options?" in prompt


def test_chat_messages_structure():
    msgs = build_chat_messages("TestBiz", "Some context", "A question?")
    assert len(msgs) == 2
    assert msgs[0]["role"] == "system"
    assert msgs[1]["role"] == "user"
    assert "TestBiz" in msgs[0]["content"]
