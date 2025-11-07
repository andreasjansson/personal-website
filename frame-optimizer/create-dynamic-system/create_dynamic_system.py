from typing import Iterator
from cog import Input, Path, BaseModel
from cog.ext.pipelines import include

claude = include("anthropic/claude-3.7-sonnet")
train_dynamic_system = include("andreasjansson/train-dynamic-system")

class ModelOutput(BaseModel):
    visualization: Path
    cycle_images: list[Path] | None = None
    cycle_losses_target1: list[float] | None = None
    cycle_losses_target2: list[float] | None = None
    historical_losses: list[float] | None = None
    model_state: Path | None = None
    optimizer_state: Path | None = None
    animation: Path | None = None
    code: str | None = None

def create_dynamic_system(
    target1: Path,
    target2: Path,
    max_attempts: int = Input(ge=1, default=20),
    width: int = Input(default=128),
    height: int = Input(default=128),
    training_steps: int = Input(default=1000),
    max_variables: int = Input(default=100),
    cycle_length: int = Input(default=20),
    loss_cycles: int = Input(default=4),
    total_cycles: int = Input(default=6),
    learning_rate: float = Input(default=0.005),
    yield_every: int = Input(default=250),
    return_animation: bool = Input(default=True),
    train_timeout: int = Input(default=600),
) -> Iterator[ModelOutput]:
    final_output: ModelOutput | None = None

    for attempt in range(max_attempts):
        print(f"Starting attempt {attempt + 1}/{max_attempts}")

        system_prompt = make_system_prompt(max_variables=max_variables)


def make_system_prompt(*, max_variables: int) -> str:
