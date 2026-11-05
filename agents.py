import json
from typing import Any, Dict, List, Literal, Optional

from groq import Groq
from pydantic import BaseModel, Field, ValidationError

MODEL_ID = "openai/gpt-oss-20b"
client = Groq()  # Reads GROQ_API_KEY from environment


class QAResult(BaseModel):
    passed: bool
    issues: List[str] = Field(default_factory=list)
    score: int = Field(ge=0, le=100)
    approved_content: str = ""


Step = Literal["research", "content", "optimization", "quality", "distribution"]


def _format_prompt(template: str, state: Dict[str, Any]) -> str:
    """Replace known {variables}; leave unknown braces untouched."""
    values = {
        "topic": state.get("topic", ""),
        "research": state.get("research", ""),
        "content": state.get("content", ""),
        "optimized": state.get("optimized", ""),
        "issues": "\n".join(state.get("qa_issues", [])),
        "qa_checklist": "\n".join(f"- {x}" for x in state.get("qa_checklist", [])),
        "language": state.get("language", "English"),
        "channels": ", ".join(state.get("channels", [])),
        "previous_attempt": state.get("previous_attempt", ""),
    }
    # Simple named replacement avoids str.format errors from braces in user text.
    for key, value in values.items():
        template = template.replace("{" + key + "}", str(value))
    return template


def run_step(step: Step, profile: dict, state: Dict[str, Any]) -> Any:
    """
    Run one pipeline step using the selected domain profile.
    For `quality`, return a validated QAResult. Other steps return text.
    """
    if step not in profile["prompts"]:
        raise ValueError(f"Step '{step}' is missing from domain profile.")

    system_prompt = profile["prompts"][step]
    user_prompt = _format_prompt(profile.get("step_inputs", {}).get(step, "{topic}"), state)

    if step == "quality":
        system_prompt += (
            "\nReturn ONLY valid JSON matching this schema: "
            '{"passed": boolean, "issues": [string, ...], '
            '"score": integer from 0 to 100, "approved_content": string}. '
            "Do not wrap JSON in markdown fences."
        )
        response = client.chat.completions.create(
            model=MODEL_ID,
            max_tokens=1200,
            temperature=0.2,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )
        raw = response.choices[0].message.content or "{}"
        try:
            parsed = json.loads(raw)
            return QAResult.model_validate(parsed)
        except (json.JSONDecodeError, ValidationError) as exc:
            # Retry exactly once for malformed QA output.
            retry = client.chat.completions.create(
                model=MODEL_ID,
                max_tokens=1200,
                temperature=0,
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": (
                        "Your previous response was invalid. Fix it and return only valid "
                        f"JSON matching the required schema. Validation error: {exc}\n"
                        f"Previous response:\n{raw}\n\nTask:\n{user_prompt}"
                    )},
                ],
            )
            retry_raw = retry.choices[0].message.content or "{}"
            return QAResult.model_validate_json(retry_raw)

    response = client.chat.completions.create(
        model=MODEL_ID,
        max_tokens=1500,
        temperature=0.5,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    )
    return response.choices[0].message.content or ""
