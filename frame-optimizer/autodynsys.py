import json
from datetime import datetime
from typing import Any
import time
import base64
import tempfile
from pathlib import Path
import re
import numpy as np
from PIL import Image
import traceback
import anthropic
from anthropic import RateLimitError
from google import genai
from google.genai.types import GenerateContentConfig

from train_dynamic_system import *
from sample_data import create_target_images, flux_target_images

MAX_VARIABLES = 100

SYSTEM_PROMPT_PRELUDE = """You're an agent with the task of creating interesting dynamic systems that produce images.

I want to create an iterated dynamic system that when iterating, oscillates between two different images. The images are inputs to the system, and I want to optimize the variables of the dynamic system with gradient descent, so every operation of the dynamic system needs to be differentiable. The oscillation should happen because of the interaction between variables at each step.

The beauty of dynamic systems is that they have very few variables, but when iterated through a step function, they produce very complex and interesting behaviours, or in our 2D case, images."""

CODE_BLOCK_REMINDER = "Remember to think before returning code, and wrap all code in <code></code> so that I can parse it out programmatically and eval() it. Don't use backticks as I will eval() everything inside the <code></code> block."


class Failed(Exception):
    pass


def generate_and_train_dynamic_system(
    anthropic_client: anthropic.Anthropic,
    code_client: genai.Client | anthropic.Client,
    num_attempts: int = 100,
    width: int = 128,
    height: int = 128,
    training_steps: int = 2000,
) -> str:
    print("starting...")

    messages = []

    #target1, target2 = create_target_images(width, height)
    target1, target2 = flux_target_images(width, height)
    target1_path = save_image_tensor(target1)
    target2_path = save_image_tensor(target2)

    for attempt in range(num_attempts):
        if not messages or len(messages) > 30:
            print("Resetting to initial messages")
            messages = [
                {
                    "role": "user",
                    "content": f"""Generate a dynamic system that implements DynamicSystem.

{CODE_BLOCK_REMINDER}""",
                }
            ]

        try:
            print(f"Starting attempt {attempt + 1}/{num_attempts}")
            is_good, code = run_attempt(
                anthropic_client,
                code_client,
                width,
                height,
                training_steps,
                messages,
                target1_path,
                target2_path,
            )
            if is_good:
                return code
        except Exception:
            error_traceback = traceback.format_exc()
            print("Caught unexpected error")
            print(error_traceback)
            messages = []

    raise Exception("Maximum attempts reached. Could not fix the system.")


def run_attempt(
    anthropic_client,
    code_client,
    width,
    height,
    training_steps,
    messages,
    target1_path,
    target2_path,
) -> tuple[bool, str]:
    system_prompt = make_system_prompt()

    code = generate_code(code_client, system_prompt, messages)
    messages.append({"role": "assistant", "content": code})

    try:
        for output in train_dynamic_system(
            code=code,
            target1=target1_path,
            target2=target2_path,
            width=width,
            height=height,
            training_steps=training_steps,
            max_variables=MAX_VARIABLES,
            cycle_length=12,
            loss_cycles=4,
            total_cycles=6,
            learning_rate=0.005,
            seed=None,
        ):
            print(
                f"Cycle losses - Target 1: {output.cycle_losses_target1}, Target 2: {output.cycle_losses_target2}"
            )
            if output.visualization:
                img = Image.open(output.visualization)
                plt.figure(figsize=(7, 7))
                plt.imshow(np.array(img))
                plt.axis("off")
                plt.show()
            final_output = output

    except Exception:
        error_traceback = traceback.format_exc()
        print(error_traceback)

        fix_prompt = f"""
I tried to use your DynamicSystem implementation but encountered an error:

{error_traceback}

{CODE_BLOCK_REMINDER}"""

        messages.append({"role": "user", "content": fix_prompt})
        return False, code

    print(f"System successfully trained")

    is_good, critique = critique_system(
        anthropic_client, final_output, target1_path, target2_path
    )
    if is_good:
        return True, code

    print(f"Not a good system: {critique}")

    fix_prompt = f"""
After looking at the outputs, I realized it's not a good system because:

{critique}

{CODE_BLOCK_REMINDER}"""

    messages.append({"role": "user", "content": fix_prompt})

    return False, code


def make_system_prompt():
    current_dir = Path(__file__).parent
    train_dynamic_system_path = current_dir / "train-dynamic-system/train_dynamic_system.py"
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
* AssertionError: The model has NNN variables (including number of elements in tensors), maximum is {MAX_VARIABLES}
* TypeError: cannot assign 'torch.cuda.FloatTensor' as parameter 'XXX' (torch.nn.Parameter or None expected)
* Immediate collapse to a single color.

Important:
* Make your system concise with <{MAX_VARIABLES} parameters (or total elements of parameter tensors).
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


def save_messages_history(messages: list[dict[str, Any]]) -> None:
    """Save the messages history to a JSON file with timestamp."""
    history_file = Path("autodynsys-messages-history.json")

    entry = {"timestamp": datetime.now().isoformat(), "messages": messages}

    if history_file.exists():
        try:
            with open(history_file, "r") as f:
                history = json.load(f)
        except json.JSONDecodeError:
            history = []
    else:
        history = []

    history.append(entry)

    with open(history_file, "w") as f:
        json.dump(history, f, indent=2)


def generate_code(
    client: anthropic.Client | genai.Client,
    system_prompt: str,
    messages: list[dict],
    attempts_left=3,
    temperature=0.8,
    thinking=True,
) -> str:
    if attempts_left == 0:
        raise Failed()

    save_messages_history(messages)

    if isinstance(client, anthropic.Client):
        full_response = claude_get_text(
            client=client,
            messages=messages,
            max_tokens=4000,
            temperature=temperature,
            system_prompt=system_prompt,
            thinking=thinking,
        )
    else:
        full_response = gemini_get_text(
            client=client,
            messages=messages,
            system_prompt=system_prompt,
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
            return generate_code(
                client, system_prompt, messages, attempts_left=attempts_left - 1
            )

        return code
    else:
        # response is just code
        last_line = full_response.strip().splitlines()
        if full_response.startswith("class") and "return " in last_line:
            return full_response

        print("Failed to generate code with a <code></code> block, trying again")
        return generate_code(
            client, system_prompt, messages, attempts_left=attempts_left - 1
        )


def critique_alternation(
    output: Output,
) -> tuple[bool, str]:
    alternation_is_good = True
    cycle_critique = []

    losses_target1 = output.cycle_losses_target1
    losses_target2 = output.cycle_losses_target2

    for i in range(len(losses_target1)):
        if i % 2 == 0 and losses_target1[i] > losses_target2[i]:
            alternation_is_good = False
            cycle_critique.append(
                f"Cycle {i + 1} should be closer to target1 but was closer to target2 "
                f"(loss_target1={losses_target1[i]:.4f}, loss_target2={losses_target2[i]:.4f})"
            )
        elif i % 2 == 1 and losses_target2[i] > losses_target1[i]:
            alternation_is_good = False
            cycle_critique.append(
                f"Cycle {i + 1} should be closer to target2 but was closer to target1 "
                f"(loss_target1={losses_target1[i]:.4f}, loss_target2={losses_target2[i]:.4f})"
            )

    if not alternation_is_good:
        critique_text = f"""After training, it failed to produce alternating patterns.

{"\n".join(cycle_critique)}"""
        return False, critique_text

    return True, ""


def critique_with_ai(
    client: anthropic.Anthropic,
    output: Output,
    target1_path: Path,
    target2_path: Path,
) -> tuple[bool, str]:
    cycle_paths = []
    for i, img_path in enumerate(output.cycle_images):
        cycle_path = Path("/tmp") / f"cycle_{i + 1}.png"
        img = Image.open(img_path)
        img.save(cycle_path)
        cycle_paths.append(cycle_path)

    assert len(cycle_paths) == 6

    all_files = [target1_path, target2_path] + cycle_paths

    system_prompt = f"""{SYSTEM_PROMPT_PRELUDE}

I have a dynamic system that I've trained to oscillate between two target images. I want you to critique how good the system is.

I want a system that produces vivid, crisp, colorful images that are similar to the target images."""

    prompt = f"""
The first two images are:
- target1.png: The first target pattern
- target2.png: The second target pattern

The remaining six images show what our dynamic system produces at the end of each cycle:
- cycle_1.png: End of cycle 1
- cycle_2.png: End of cycle 2
- cycle_3.png: End of cycle 3
- cycle_4.png: End of cycle 4
- cycle_5.png: End of cycle 5
- cycle_6.png: End of cycle 6

Please critique:
1. Are the generated images vivid and colorful?
2. Are they perceptually similar to the targets? (odd cycles to Target 1, even cycles to Target 2)
3. Overall, is this a good dynamic system?

Be critical and thorough in your assessment.

Start with YES/NO, followed by reasoning, since I will use your initial word (YES or NO) in a programmatic context.
        """

    content = []
    for file_path in all_files:
        with open(file_path, "rb") as f:
            encoded_string = base64.b64encode(f.read()).decode()

        mime_type = "image/png"

        content.append(
            {
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": mime_type,
                    "data": encoded_string,
                },
            }
        )

    content.append({"type": "text", "text": prompt})

    critique = claude_get_text(
        client,
        system_prompt=system_prompt,
        messages=[{"role": "user", "content": content}],
        temperature=0,
        max_tokens=1000,
    )

    is_good = critique.lower().startswith("yes")

    return is_good, critique


def critique_system(
    client: anthropic.Anthropic,
    output: Output,
    target1_path: Path,
    target2_path: Path,
) -> tuple[bool, str]:
    alternation_is_good, alternation_critique = critique_alternation(output)
    ai_is_good, ai_critique = critique_with_ai(
        client, output, target1_path, target2_path
    )

    is_good = alternation_is_good and ai_is_good
    critique = "\n\n".join([alternation_critique, ai_critique])
    return is_good, critique


def claude_get_text(
    client: anthropic.Client,
    system_prompt: str,
    messages: list[dict],
    temperature: float,
    max_tokens: int,
    attempts_left=3,
    thinking=False,
    thinking_tokens=2000,
) -> str:
    if attempts_left == 0:
        raise Failed("Rate limited by Anthropic too many times")

    if thinking:
        thinking_dict = {"type": "enabled", "budget_tokens": thinking_tokens}
        temperature = 1
    else:
        thinking_dict = {"type": "disabled"}
        thinking_tokens = 0

    try:
        response = client.messages.create(
            model="claude-3-7-sonnet-latest",
            max_tokens=max_tokens + thinking_tokens,
            temperature=temperature,
            system=system_prompt,
            messages=messages,
            thinking=thinking_dict,
        )
    except RateLimitError:
        print("Rate limited by Anthropic, sleeping for 60 seconds")
        time.sleep(60)
        return claude_get_text(
            client,
            system_prompt,
            messages,
            temperature,
            max_tokens,
            thinking=thinking,
            thinking_tokens=thinking_tokens,
            attempts_left=attempts_left - 1,
        )

    if thinking:
        for c in response.content:
            if c.type != "text" and hasattr(c, "thinking"):
                print(f"THINKING\n{c.thinking}")

    return response.content[-1].text


def gemini_get_text(
    client: genai.Client,
    system_prompt: str,
    messages: list[dict],
    temperature: float,
):
    messages = [message_to_gemini(m) for m in messages]

    return client.models.generate_content(
        model="gemini-2.5-pro-exp-03-25",
        contents=messages,
        config=GenerateContentConfig(system_instruction=system_prompt, temperature=1.0),
    ).text


def message_to_gemini(msg):
    role = msg["role"]
    return {
        "role": role if role == "user" else "model",
        "parts": [{"text": msg["content"]}],
    }


def save_image_tensor(tensor: torch.Tensor) -> Path:
    temp_file = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
    filename = Path(temp_file.name)
    temp_file.close()

    img_array = (tensor.cpu().numpy() * 255).astype(np.uint8)
    img = Image.fromarray(img_array)
    img.save(filename)

    return filename
