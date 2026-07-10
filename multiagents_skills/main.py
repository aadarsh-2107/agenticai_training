import os
from typing import Callable, TypedDict

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.tools import tool

# create_agent + middleware primitives (LangChain agents).
from langchain.agents import create_agent
from langchain.agents.middleware import (
    AgentMiddleware,
    ModelRequest,
    ModelResponse,
)
from langgraph.checkpoint.memory import InMemorySaver

load_dotenv()

# region Prompt
system_prompt = """
You are a helpful Corporate Policy Assistant that helps employees understand and
apply corporate policy in their work.
You can answer questions about policy and provide guidance on how to comply with them.
"""
# endregion Prompt


# region Skill
class Skill(TypedDict):
    name: str
    description: str
    path: str


BASE_DIR = "multiagents_skills/skills"

SKILLS: list[Skill] = [
    {
        "name": "hr_policy",
        "description": "Human Resources policies and procedures.",
        "path": f"{BASE_DIR}/hr_policy.md",
    },
    {
        "name": "travel_policy",
        "description": "Travel related policies and procedures.",
        "path": f"{BASE_DIR}/travel_policy.md",
    },
    {
        "name": "compliance_policy",
        "description": "Corporate compliance policies and procedures.",
        "path": f"{BASE_DIR}/compliance_policy.md",
    },
]
# endregion Skill


# region Model
llm = ChatOpenAI(model=os.getenv("OPENAI_MODEL", "gpt-5.5"), temperature=0)
# endregion Model


# region Tools
@tool
def load_skill(skill_name: str) -> str:
    """
    Load the full content of the skill into the agent's context.

    Use this when you need to handle a specific type of request. It provides
    comprehensive instructions, policies, and guidelines for the skill area.

    Args:
        skill_name: The name of the skill to load, e.g. hr_policy,
            travel_policy, or compliance_policy.
    """
    for skill in SKILLS:
        if skill["name"] == skill_name:
            try:
                with open(skill["path"], "r", encoding="utf-8") as f:
                    content = f.read()
            except FileNotFoundError:
                return f"Skill '{skill_name}' file not found at {skill['path']}."
            return f"Loaded skill: {skill_name}\n\n{content}"
    return f"{skill_name} is not available"
# endregion Tools


# region Middleware
class SkillMiddleware(AgentMiddleware):
    """Middleware to manage skill loading and context for the agent."""

    tools = [load_skill]

    def __init__(self) -> None:
        super().__init__()
        skill_list = [
            f"  - **{skill['name']}**: {skill['description']}" for skill in SKILLS
        ]
        self.skills_prompt = "\n".join(skill_list)

    def wrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], ModelResponse],
    ) -> ModelResponse:
        """Inject the list of available skills into the system prompt."""
        skills_addendum = (
            f"\n\nAvailable skills:\n{self.skills_prompt}\n\n"
            "Use the load_skill tool to load a policy's full content when needed, "
            "then answer questions about the policy and how to comply with it."
        )
        base = request.system_prompt or system_prompt
        new_prompt = base + skills_addendum
        # `.override(...)` returns a modified copy on recent LangChain versions.
        # Fall back to mutating the attribute directly on older versions.
        if hasattr(request, "override"):
            request = request.override(system_prompt=new_prompt)
        else:
            request.system_prompt = new_prompt
        # A wrap_model_call middleware MUST call handler() and return its
        # ModelResponse. Returning None causes:
        #   AttributeError: 'NoneType' object has no attribute 'result'
        return handler(request)
# endregion Middleware


# region Agent
agent = create_agent(
    model=llm,
    system_prompt=system_prompt,
    middleware=[SkillMiddleware()],
    checkpointer=InMemorySaver(),
)
# endregion Agent


if __name__ == "__main__":
    user_prompt = "Can you please help with the company travel policies?"
    config = {"configurable": {"thread_id": "my_skill_thread"}}

    result = agent.invoke(
        {"messages": [{"role": "user", "content": user_prompt}]},
        config=config,
    )

    print(result["messages"][-1].content)
