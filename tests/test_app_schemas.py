import pytest
from pydantic import ValidationError

from app.schemas import AgentState, Message, ToolCall, ToolResult, UserTurn


def test_user_turn_has_safe_defaults():
    turn = UserTurn(user_id="u-1", message="What is the weather?")

    assert turn.tenant == "lab"
    assert turn.priority == 0
    assert turn.mode == "interactive"
    assert turn.max_steps == 4


@pytest.mark.parametrize(
    "payload",
    [
        {"user_id": "u-1", "message": ""},
        {"user_id": "u-1", "message": "hi", "priority": 21},
        {"user_id": "u-1", "message": "hi", "timeout_s": 0},
    ],
)
def test_user_turn_rejects_invalid_admission_inputs(payload):
    with pytest.raises(ValidationError):
        UserTurn.model_validate(payload)


def test_tool_call_and_result_are_typed():
    call = ToolCall(id="call-1", name="weather", arguments={"city": "Seattle"})
    result = ToolResult(call_id=call.id, name=call.name, output={"temp_c": 12})

    assert call.arguments["city"] == "Seattle"
    assert result.ok is True


def test_agent_state_serializes_conversation_and_observation():
    turn = UserTurn(user_id="u-1", message="Use the weather tool")
    state = AgentState(
        turn=turn,
        messages=[Message(role="user", content=turn.message)],
        observations=[
            ToolResult(call_id="call-1", name="weather", output="12 C")
        ],
        step=1,
    )

    document = state.model_dump()
    assert document["turn"]["user_id"] == "u-1"
    assert document["observations"][0]["output"] == "12 C"

