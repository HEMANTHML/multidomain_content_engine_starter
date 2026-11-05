from pathlib import Path
from typing import Dict, Any

import yaml
from pydantic import BaseModel, Field, ValidationError


class DomainProfile(BaseModel):
    name: str
    description: str = ""
    tone: str
    constraints: Dict[str, Any] = Field(default_factory=dict)
    channels: list[str]
    step_labels: Dict[str, str]
    prompts: Dict[str, str]
    step_inputs: Dict[str, str] = Field(default_factory=dict)
    qa_checklist: list[str]


def load_profiles(directory: str) -> dict[str, dict]:
    """Load and validate all *.yaml profiles. Raise on any invalid profile."""
    profile_dir = Path(directory)
    if not profile_dir.exists():
        raise FileNotFoundError(f"Domain profile directory not found: {profile_dir}")

    profiles = {}
    for path in sorted(profile_dir.glob("*.yaml")):
        with path.open("r", encoding="utf-8") as stream:
            raw = yaml.safe_load(stream) or {}
        try:
            validated = DomainProfile.model_validate(raw)
        except ValidationError as exc:
            raise ValueError(f"Invalid domain profile '{path.name}':\n{exc}") from exc
        profile_id = path.stem
        profiles[profile_id] = validated.model_dump()

    if not profiles:
        raise ValueError(f"No domain YAML profiles found in {profile_dir}")
    return profiles


def get_profile(domain_id: str, profiles: dict[str, dict]) -> dict:
    try:
        return profiles[domain_id]
    except KeyError as exc:
        raise ValueError(f"Domain '{domain_id}' is not configured.") from exc
