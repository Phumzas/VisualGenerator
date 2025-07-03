import gradio as gr
import numpy as np
import random
from diffusers import DiffusionPipeline
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"
model_repo_id = "stabilityai/sdxl-turbo"  # You can change model here

torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

pipe = DiffusionPipeline.from_pretrained(model_repo_id, torch_dtype=torch_dtype)
pipe = pipe.to(device)

MAX_SEED = np.iinfo(np.int32).max
MAX_IMAGE_SIZE = 1024

def infer(
    prompt,
    negative_prompt,
    seed,
    randomize_seed,
    width,
    height,
    guidance_scale,
    num_inference_steps,
    progress=gr.Progress(track_tqdm=True),
):
    if randomize_seed:
        seed = random.randint(0, MAX_SEED)

    generator = torch.Generator(device=device).manual_seed(seed)

    image = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt if negative_prompt else None,
        guidance_scale=guidance_scale,
        num_inference_steps=num_inference_steps,
        width=width,
        height=height,
        generator=generator,
    ).images[0]

    return image, seed

examples = [
    "An astronaut riding a green horse",
    "Sneakers on a white background, in a photorealistic style, with a bright mood",
    "Little cat reading under a tree, in a cartoon style, with a happy mood",
    "The human brain with highlighted neurons, in an abstract style, with a calm mood",
    "Spaceship landing on an alien world, in a digital painting style, with a futuristic mood",
    "Smartwatch UI concept, in a 3D render style, with a vibrant mood",
    "A delicious ceviche cheesecake slice",
]

css = """
#col-container {
    margin: 0 auto;
    max-width: 640px;
}
"""

with gr.Blocks(css=css) as demo:
    with gr.Column(elem_id="col-container"):
        gr.Markdown(" # Text-to-Image Gradio Template")

        with gr.Row():
            prompt = gr.Textbox(
                label="Prompt",
                placeholder="Enter your prompt",
                lines=1,
                max_lines=1,
                container=False,
            )
            run_button = gr.Button("Run", scale=0, variant="primary")

        result = gr.Image(label="Result")

        with gr.Accordion("Advanced Settings", open=False):
            negative_prompt = gr.Textbox(
                label="Negative prompt",
                placeholder="Enter a negative prompt (optional)",
                lines=1,
                max_lines=1,
            )
            seed = gr.Slider(
                label="Seed",
                minimum=0,
                maximum=MAX_SEED,
                step=1,
                value=0,
            )
            randomize_seed = gr.Checkbox(label="Randomize seed", value=True)

            with gr.Row():
                width = gr.Slider(
                    label="Width",
                    minimum=256,
                    maximum=MAX_IMAGE_SIZE,
                    step=64,
                    value=1024,
                )
                height = gr.Slider(
                    label="Height",
                    minimum=256,
                    maximum=MAX_IMAGE_SIZE,
                    step=64,
                    value=1024,
                )

            with gr.Row():
                guidance_scale = gr.Slider(
                    label="Guidance scale",
                    minimum=1.0,
                    maximum=20.0,
                    step=0.1,
                    value=7.5,
                )
                num_inference_steps = gr.Slider(
                    label="Number of inference steps",
                    minimum=1,
                    maximum=50,
                    step=1,
                    value=25,
                )

        gr.Examples(examples=examples, inputs=prompt)

        # Bind events
        run_button.click(
            fn=infer,
            inputs=[
                prompt,
                negative_prompt,
                seed,
                randomize_seed,
                width,
                height,
                guidance_scale,
                num_inference_steps,
            ],
            outputs=[result, seed],
        )
        prompt.submit(
            fn=infer,
            inputs=[
                prompt,
                negative_prompt,
                seed,
                randomize_seed,
                width,
                height,
                guidance_scale,
                num_inference_steps,
            ],
            outputs=[result, seed],
        )

if __name__ == "__main__":
    demo.launch()

