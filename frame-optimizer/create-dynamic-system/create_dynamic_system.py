import tempfile
import urllib.request
import os
import re
from dataclasses import dataclass
from cog import Input, Path, BaseModel, include
from replicate.exceptions import ModelError

claude = include("anthropic/claude-3.7-sonnet")
train_dynamic_system = include("andreasjansson/train-dynamic-system")

SYSTEM_PROMPT_PRELUDE = """You're an agent with the task of creating interesting dynamic systems that produce images.

I want to create an iterated dynamic system that when iterating, oscillates between two different images. The images are inputs to the system, and I want to optimize the variables of the dynamic system with gradient descent, so every operation of the dynamic system needs to be differentiable. The oscillation should happen because of the interaction between variables at each step.

The beauty of dynamic systems is that they have very few variables, but when iterated through a step function, they produce very complex and interesting behaviours, or in our 2D case, images."""


class ModelOutput(BaseModel):
    visualization: Path
    cycle_images: list[Path]
    cycle_losses_target1: list[float]
    cycle_losses_target2: list[float]
    historical_losses: list[float]
    model_state: Path
    optimizer_state: Path
    animation: Path
    code: str
    critique: str


@dataclass
class FailedAttempt:
    code: str
    logs: str


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
    return_animation: bool = Input(default=True),
    train_timeout: int = Input(default=600),
) -> ModelOutput:
    failed_attempts = []
    for attempt in range(max_attempts):
        print(f"Starting attempt {attempt + 1}/{max_attempts}")

        system_prompt = make_system_prompt(max_variables=max_variables)
        code = generate_code(system_prompt, failed_attempts)

        try:
            output_dicts = train_dynamic_system(
                code=code,
                width=width,
                height=height,
                target1=target1,
                target2=target2,
                timeout=train_timeout,
                loss_cycles=loss_cycles,
                yield_every=training_steps,
                cycle_length=cycle_length,
                total_cycles=total_cycles,
                learning_rate=learning_rate,
                max_variables=max_variables,
                training_steps=training_steps,
                return_animation=return_animation,
            )
            output_dict = output_dicts[-1]
            visualization = download(output_dict["visualization"])
            critique = critique_system(visualization)
            output = yielded_dict_to_output(
                output_dict, visualization, code=code, critique=critique
            )
            print("Successfully trained model")
            return output

        except ModelError as e:
            logs = e.prediction.logs
            print(f"Failed to train model:\n{logs}")
            failed_attempts.append(FailedAttempt(code=code, logs=logs))

    raise Exception(
        f"Failed to create a working dynamic system after {max_attempts} attempts"
    )


# Only works for final, complete output
def yielded_dict_to_output(
    yielded_dict: dict, visualization: Path, code: str, critique: str
) -> ModelOutput:
    assert "animation" in yielded_dict

    output = ModelOutput(
        visualization=visualization,
        cycle_images=download_list(yielded_dict["cycle_images"]),
        cycle_losses_target1=yielded_dict["cycle_losses_target1"],
        cycle_losses_target2=yielded_dict["cycle_losses_target2"],
        historical_losses=yielded_dict["historical_losses"],
        model_state=download(yielded_dict["model_state"]),
        optimizer_state=download(yielded_dict["optimizer_state"]),
        animation=download(yielded_dict["animation"]),
        code=code,
        critique=critique,
    )

    return output


def generate_code(
    system_prompt: str,
    failed_attempts: list[FailedAttempt],
    attempts_left=3,
    temperature=0.8,
) -> str:
    if attempts_left == 0:
        raise Exception("Failed to generate code")

    prompt = make_prompt(failed_attempts)
    full_response = claude(
        system_prompt=system_prompt,
        prompt=prompt,
        temperature=temperature,
    )

    assert full_response
    print(full_response)

    code_match = re.search(
        r"(?:<code>(.*?)</code>|```(?:\s*python\s*)?\n(.*?)\n```)",
        full_response,
        re.DOTALL,
    )
    if code_match:
        if code_match.group(1):
            code = code_match.group(1).strip()
        else:
            code = code_match.group(2).strip()

        if not code:
            print("Failed to generate code with a <code></code> block, trying again")
            return generate_code(
                system_prompt, failed_attempts, attempts_left=attempts_left - 1
            )

        return code
    else:
        # response is just code
        last_line = full_response.strip().splitlines()
        if full_response.startswith("class") and "return " in last_line:
            return full_response

        print("Failed to generate code with a <code></code> block, trying again")
        return generate_code(
            system_prompt, failed_attempts, attempts_left=attempts_left - 1
        )


def make_prompt(failed_attempts: list[FailedAttempt]) -> str:
    if failed_attempts:
        multiple_failures = len(failed_attempts) > 1
        prompt = f"You previously generated the following system{'s' if multiple_failures > 1 else ''} that threw errors:\n\n"
        for i, failed_attempt in enumerate(failed_attempts):
            if multiple_failures:
                prompt += f"Failed attempt #{i}:\n\n"
            prompt += "CODE:\n```" + failed_attempt.code + "```\n\n"
            prompt += "LOGS:\n```" + failed_attempt.logs + "```\n\n"

        if multiple_failures:
            prompt += "Create a new DynamicSystem class that fixes the errors in the last attempt, while not repeating the mistakes you made in past attempts. You might need to re-think your approach completely if you can't reason your way out of this error."
        else:
            prompt += "Create a new DynamicSystem class that fixes the errors."

    else:
        prompt = "Generate a dynamic system that implements DynamicSystem."

    prompt += "\n\nRemember to think (and write down your thinking and reasoning) before returning code, and wrap all code in <code></code> so that I can parse it out programmatically and eval() it. Don't use backticks as I will eval() everything inside the <code></code> block."

    return prompt


def make_system_prompt(*, max_variables: int) -> str:
    current_dir = Path(__file__).parent
    train_dynamic_system_path = current_dir / "train_dynamic_system.py"
    train_dynamic_system_py_contents = train_dynamic_system_path.read_text()

    return f"""{SYSTEM_PROMPT_PRELUDE}

While well-known iterated systems like Bogdanov maps, Chirikov–Taylor maps, Duffing maps, Coupled Differentiable Maps, Continuous Differentiable Cellular Automata, and Coupled Reaction-Diffusion Equations offer valuable insights into generating complex 2D images from simple rules, I want you to push further and synthesize ideas from a wider range of dynamic phenomena. Draw inspiration from areas such as:
* Physics, Chemistry & Complex Systems: Consider principles from phase transitions, wave interference, fluid dynamics, chemical oscillators... Also, explore how elements from chaotic dynamics (like sensitivity to conditions, non-linear feedback loops, bifurcation structures, or strange attractors) could be adapted. The goal isn't necessarily a purely chaotic output, but perhaps these mechanisms can create highly complex, interesting transition dynamics between the target states A and B, or be modulated by the system's state relative to A/B to enforce the oscillation.
* Biological Pattern Formation & Ecology: Think morphogenesis (Turing patterns), neural network dynamics (attractor states, oscillations), ecological cycles (predator-prey dynamics between 'A-ness' and 'B-ness'), or genetic regulatory network motifs.
* Modern AI & Computational Models: Look at concepts from differentiable physics, Neural ODEs, continuous Cellular Automata like Lenia, generative model internal dynamics, or even abstract ideas from swarm intelligence or self-organization.
* Geometric & Topological Dynamics: Could the evolution involve changing shapes, boundaries, or connectivity in interesting ways?

Be creative in how you implement the oscillation mechanism itself. Instead of an explicit counter, how can the system's internal state, spatial interactions, and relationship to the target images (A and B) intrinsically drive the switching behavior? Perhaps explore:
* Systems with adaptive thresholds or boundaries (e.g., sigmoids whose parameters evolve).
* Dynamics based on competitive exclusion or resource competition between elements favoring A vs. B.
* Mechanisms involving hysteresis or memory embedded within the state variables.
* Spatial wave propagation or signaling that triggers state shifts.
* Think freely about combining these diverse inspirations, ensuring the core requirements (iterative, differentiable, image output, two-target oscillation) are met.
* Your implementation should implement the DynamicSystem abstract class and will be optimized with the DynamicSystemTrainer (see code below).
* I want a system that produces vivid, crisp, colorful images.

Your implementation must be differentiable, which can be non-trivial in a dynamic system. Your implementation should also produce an interesting image with multiple colors even at step 0.

Common errors to avoid include:
* RuntimeError: Trying to backward through the graph a second time (or directly access saved tensors after they have already been freed). Saved intermediate values of the graph are freed when you call .backward() or autograd.grad(). Specify retain_graph=True if you need to backward through the graph a second time or if you need to access saved tensors after calling backward.
* RuntimeError: one of the variables needed for gradient computation has been modified by an inplace operation
* RuntimeError: a view of a leaf Variable that requires grad is being used in an in-place operation.
* NotImplementedError: Only 2D, 3D, 4D, 5D padding with non-constant padding are supported for now
* RuntimeError: Expected all tensors to be on the same device, but found at least two devices, cuda:0 and cpu!
* AssertionError: The model has NNN variables (including number of elements in tensors), maximum is {max_variables}
* TypeError: cannot assign 'torch.cuda.FloatTensor' as parameter 'XXX' (torch.nn.Parameter or None expected)
* Immediate collapse to a single color.

Important:
* Make your system concise with <{max_variables} parameters (or total elements of parameter tensors).
* Don't include many comments.
* Give the class a descriptive name and a one or two sentence description in a class comment.
* Cleanly separate parameters from state variables that are updated in step(). State variables should not be trainable parameters, and trainable parameters should not update.
* reset_state() should initialize all state variables (give them descriptive names, not just a single self.state), and should be called at the end of __init__
* Don't explicitly include cycle lengths in the code, since the cycle length should be implicit in the final trained model
* Avoid using an explicit (step) counter in your code -- the step() function should ideally implement some interaction between the variables
* Don't use hard decisions (e.g. modulo operators) since everything needs to be differentiable.
* Don't modify .data directly as this breaks the differentiation graph.
* The __init__() method of the DynamicSystem class should take no arguments

And extra important:
* Think before returning code, and wrap all code in <code></code> so that I can parse it out programmatically and eval() it. Don't use backticks as I will eval() everything inside the <code></code> block.

Below follows my existing code, in which I will use the dynamic system class you output.

{train_dynamic_system_py_contents}
"""


def critique_system(visualization: Path):
    system_prompt = f"""{SYSTEM_PROMPT_PRELUDE}

    I have a dynamic system that I've trained to oscillate between two target images. I want you to critique how good the system is.

I want a system that produces vivid, crisp, colorful images that are similar to the target images."""

    prompt = """The first two images are:
- Target 1: The first target pattern
- Target 2: The second target pattern

The remaining six images show what our dynamic system produces at the end of each cycle:
- Cycle 1: End of cycle 1
- Cycle 2: End of cycle 2
- Cycle 3: End of cycle 3
- Cycle 4: End of cycle 4
- Cycle 5: End of cycle 5
- Cycle 6: End of cycle 6

There is also a loss curve over historical losses when training the system, and a curve of losses over the iterated steps for the trained model.

Please critique:
1. Are the generated images vivid and colorful?
2. Are they perceptually similar to the targets? (odd cycles to Target 1, even cycles to Target 2)
3. Overall, is this a good dynamic system?

Be critical and thorough in your assessment.

Start with YES/NO, followed by reasoning, since I will use your initial word (YES or NO) in a programmatic context."""

    return claude(
        prompt=prompt,
        system_prompt=system_prompt,
        image=visualization,
        max_image_resolution=2,
    )


def download(url: str) -> Path:
    extension = os.path.splitext(url)[1]
    fd, temp_path = tempfile.mkstemp(suffix=extension)
    os.close(fd)

    urllib.request.urlretrieve(url, temp_path)
    return Path(temp_path)


def download_list(urls: list[str]) -> list[Path]:
    return [path for path in [download(url) for url in urls] if path]
