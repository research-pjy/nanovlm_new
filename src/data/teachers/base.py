"""Provider-neutral teacher boundary; records never depend on a provider class."""
from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True)
class TeacherOutput:
    text: str
    truncated: bool = False


class TeacherGenerator(Protocol):
    @property
    def identity(self) -> dict:
        """JSON-serializable model and runtime provenance used to guard resume."""
        ...

    def generate_batch(self, prompts: list[str], seed: int) -> list[TeacherOutput]:
        ...

    def memory_stats(self) -> dict:
        ...
