"""Tests for uwcopilot.llm.fake_client: deterministic, offline, always labeled as fake."""

import pytest

from uwcopilot.llm.base import LLMClient, LLMMessage
from uwcopilot.llm.fake_client import FakeLLM, FakeLLMExhausted

USER = [LLMMessage(role="user", content="What is the DSCR minimum?")]


def test_scripted_responses_come_back_in_order() -> None:
    fake = FakeLLM(["first answer", "second answer"])
    assert fake.complete("system", USER, max_tokens=50).text == "first answer"
    assert fake.complete("system", USER, max_tokens=50).text == "second answer"
    with pytest.raises(FakeLLMExhausted):
        fake.complete("system", USER, max_tokens=50)


def test_output_is_labeled_fake_and_free() -> None:
    response = FakeLLM(["anything"]).complete("system", USER, max_tokens=50)
    assert response.is_fake
    assert response.provider == "fake"
    assert response.model.startswith("fake-")
    assert response.cost_usd == 0.0
    assert response.cached is False


def test_fake_model_name_must_stay_labeled() -> None:
    with pytest.raises(ValueError):
        FakeLLM(["x"], model="claude-haiku-5-5")


def test_callable_responder_sees_the_prompt() -> None:
    fake = FakeLLM(lambda system, messages: f"{system}|{messages[-1].content}")
    response = fake.complete("SYS", USER, max_tokens=50)
    assert response.text == "SYS|What is the DSCR minimum?"


def test_calls_are_recorded_and_token_counts_are_deterministic() -> None:
    fake = FakeLLM(["two words"])
    response = fake.complete("one", USER, max_tokens=7, temperature=0.5)
    assert fake.calls == [{"system": "one", "messages": USER, "max_tokens": 7, "temperature": 0.5}]
    assert response.output_tokens == 2
    assert response.input_tokens == len("oneWhat is the DSCR minimum?".split())


def test_fake_satisfies_the_client_protocol() -> None:
    assert isinstance(FakeLLM(["x"]), LLMClient)
