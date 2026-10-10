"""Typed application contracts for the tool-using agent.

These models describe the boundary between a client/agent loop and the
serving system.  They are deliberately independent of vLLM's internal
request and scheduling types.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


Role = Literal["system", "user", "assistant", "tool"]


class Message(BaseModel):
    """One message in the growing agent conversation."""

    model_config = ConfigDict(extra="forbid")

    role: Role
    content: str = Field(min_length=1)
    name: str | None = None
    tool_call_id: str | None = None


class UserTurn(BaseModel):
    """Validated input for one interactive or batch agent turn."""

    model_config = ConfigDict(extra="forbid")

    user_id: str = Field(min_length=1)
    message: str = Field(min_length=1)
    tenant: str = Field(default="lab", min_length=1)
    priority: int = Field(default=0, ge=0, le=20)
    timeout_s: float = Field(default=30.0, gt=0)
    max_steps: int = Field(default=4, ge=1, le=32)
    mode: Literal["interactive", "batch"] = "interactive"


class ToolCall(BaseModel):
    """A model-requested tool invocation before execution."""

    model_config = ConfigDict(extra="forbid")

    id: str = Field(min_length=1)
    name: str = Field(min_length=1)
    arguments: dict[str, Any] = Field(default_factory=dict)


class ToolResult(BaseModel):
    """The application-level observation returned by a tool."""

    model_config = ConfigDict(extra="forbid")

    call_id: str = Field(min_length=1)
    name: str = Field(min_length=1)
    output: Any
    ok: bool = True


class AgentState(BaseModel):
    """State carried through think -> tool -> observe -> answer."""

    model_config = ConfigDict(extra="forbid")

    turn: UserTurn
    messages: list[Message] = Field(default_factory=list)
    pending_calls: list[ToolCall] = Field(default_factory=list)
    observations: list[ToolResult] = Field(default_factory=list)
    step: int = Field(default=0, ge=0)
    done: bool = False

