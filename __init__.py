"""Text to Image plugin.

| Copyright 2017-2023, Voxel51, Inc.
| `voxel51.com <https://voxel51.com/>`_
|
"""

from datetime import datetime
import os
import uuid
from importlib.util import find_spec
from importlib.metadata import version as mversion
from packaging import version as pversion

import fiftyone.operators as foo
from fiftyone.operators import types
import fiftyone as fo
import fiftyone.core.utils as fou
from fiftyone.core.utils import add_sys_path

import requests

openai = fou.lazy_import("openai")
replicate = fou.lazy_import("replicate")

SD_MODEL_URL = "stability-ai/stable-diffusion:27b93a2413e7f36cd83da926f3656280b2931564ff050bf9575f1fdf9bcd7478"
SD_SCHEDULER_CHOICES = (
    "DDIM",
    "K_EULER",
    "K_EULER_ANCESTRAL",
    "PNDM",
    "K-LMS",
)
SD_SIZE_CHOICES = (
    "128",
    "256",
    "384",
    "512",
    "576",
    "640",
    "704",
    "768",
    "832",
    "896",
    "960",
    "1024",
)

SDXL_MODEL_URL = "stability-ai/sdxl:c221b2b8ef527988fb59bf24a8b97c4561f1c671f73bd389f866bfb27c061316"
SDXL_SCHEDULER_CHOICES = (
    "DDIM",
    "K_EULER",
    "K_EULER_ANCESTRAL",
    "PNDM",
    "DPMSolverMultistep",
    "KarrasDPM",
    "HeunDiscrete",
)

SDXL_REFINE_CHOICES = ("None", "Expert Ensemble", "Base")

SDXL_REFINE_MAP = {
    "None": "no_refiner",
    "Expert Ensemble": "expert_ensemble_refiner",
    "Base": "base_image_refiner",
}

SDXL_LIGHTNING_MODEL_URL = "lucataco/sdxl-lightning-4step:727e49a643e999d602a896c774a0658ffefea21465756a6ce24b7ea4165eba6a"
SDXL_LIGHTNING_SIZE_CHOICES = ("1024", "1280")
SDXL_LIGHTNING_NEGATIVE_PROMPT_DEFAULT = "worst quality, low quality"
SDXL_LIGHTNING_SCHEDULER_CHOICES = (
    "DDIM",
    "DPMSolverMultistep",
    "HeunDiscrete",
    "KarrasDPM",
    "K_EULER_ANCESTRAL",
    "K_EULER",
    "PNDM",
    "DPM+2MSDE",
)
SDXL_LIGHTNING_NUM_INFERENCE_STEPS_DEFAULT = 4
SDXL_LIGHTNING_GUIDANCE_SCALE_DEFAULT = 7.5


SSD1B_MODEL_URL = "lucataco/ssd-1b:1ee85ef681d5ad3d6870b9da1a4543cb3ad702d036fa5b5210f133b83b05a780"
SSD1B_SCHEDULER_CHOICES = (
    "DDIM",
    "DDPMSolverMultistep",
    "HeunDiscrete",
    "KarrasDPM",
    "K_EULER_ANCESTRAL",
    "K_EULER",
    "PNDM",
)

PLAYGROUND_V2_MODEL_URL = "playgroundai/playground-v2-1024px-aesthetic:42fe626e41cc811eaf02c94b892774839268ce1994ea778eba97103fe1ef51b8"
PLAYGROUND_V2_SCHEDULER_CHOICES = (
    "DDIM",
    "DPMSolverMultistep",
    "HeunDiscrete",
    "K_EULER_ANCESTRAL",
    "K_EULER",
    "PNDM",
)

LC_MODEL_URL = "luosiallen/latent-consistency-model:553803fd018b3cf875a8bc774c99da9b33f36647badfd88a6eec90d61c5f62fc"

VQGAN_MODEL_URL = "mehdidc/feed_forward_vqgan_clip:28b5242dadb5503688e17738aaee48f5f7f5c0b6e56493d7cf55f74d02f144d8"

KANDINSKY_MODEL_URL = "ai-forever/kandinsky-2.2:ea1addaab376f4dc227f5368bbd8eff901820fd1cc14ed8cad63b29249e9d463"
KANDINSDKY_SIZE_CHOICES = (
    "384",
    "512",
    "576",
    "640",
    "704",
    "768",
    "960",
    "1024",
    "1152",
    "1280",
    "1536",
    "1792",
    "2048",
)


DALLE2_SIZE_CHOICES = ("256x256", "512x512", "1024x1024")

DALLE3_SIZE_CHOICES = ("1024x1024", "1024x1792", "1792x1024")

DALLE3_QUALITY_CHOICES = ("standard", "hd")


def allows_replicate_models():
    """Returns whether the current environment allows replicate models."""
    return (
        find_spec("replicate") is not None
        and "REPLICATE_API_TOKEN" in os.environ
    )


def allows_openai_models():
    """Returns whether the current environment allows openai models."""
    return find_spec("openai") is not None and "OPENAI_API_KEY" in os.environ


def allows_diffusers_models():
    """Returns whether the current environment allows diffusers models."""
    if find_spec("diffusers") is None:
        return False
    version = mversion("diffusers")
    return pversion.parse(version) >= pversion.parse("0.24.0")


def download_image(image_url, filename):
    img_data = requests.get(image_url).content
    with open(filename, "wb") as handler:
        handler.write(img_data)


class Text2Image:
    """Wrapper for a Text2Image model."""

    def __init__(self):
        self.name = None
        self.model_name = None

    def generate_image(self, ctx):
        pass


class StableDiffusion(Text2Image):
    """Wrapper for a StableDiffusion model."""

    def __init__(self):
        super().__init__()
        self.name = "stable-diffusion"
        self.model_name = SD_MODEL_URL

    def generate_image(self, ctx):
        prompt = ctx.params.get("prompt", "None provided")
        width = int(ctx.params.get("width_choices", "None provided"))
        height = int(ctx.params.get("height_choices", "None provided"))
        inference_steps = ctx.params.get("inference_steps", "None provided")
        scheduler = ctx.params.get("scheduler_choices", "None provided")

        response = replicate.run(
            self.model_name,
            input={
                "prompt": prompt,
                "width": width,
                "height": height,
                "inference_steps": inference_steps,
                "scheduler": scheduler,
            },
        )
        if type(response) == list:
            response = response[0]
        return response


class SDXL(Text2Image):
    """Wrapper for a StableDiffusion XL model."""

    def __init__(self):
        super().__init__()
        self.name = "sdxl"
        self.model_name = SDXL_MODEL_URL

    def generate_image(self, ctx):
        prompt = ctx.params.get("prompt", "None provided")
        inference_steps = ctx.params.get("inference_steps", 50.0)
        scheduler = ctx.params.get("scheduler_choices", "None provided")
        guidance_scale = ctx.params.get("guidance_scale", 7.5)
        refiner = ctx.params.get("refine_choices", SDXL_REFINE_CHOICES[0])
        refiner = SDXL_REFINE_MAP[refiner]
        refine_steps = ctx.params.get("refine_steps", None)
        negative_prompt = ctx.params.get("negative_prompt", None)
        high_noise_frac = ctx.params.get("high_noise_frac", None)

        _inputs = {
            "prompt": prompt,
            "inference_steps": inference_steps,
            "scheduler": scheduler,
            "refine": refiner,
            "guidance_scale": guidance_scale,
        }
        if negative_prompt is not None:
            _inputs["negative_prompt"] = negative_prompt
        if refine_steps is not None:
            _inputs["refine_steps"] = refine_steps
        if high_noise_frac is not None:
            _inputs["high_noise_frac"] = high_noise_frac

        response = replicate.run(
            self.model_name,
            input=_inputs,
        )
        if type(response) == list:
            response = response[0]
        return response


class SDXLLightning(Text2Image):
    """Wrapper for a StableDiffusion XL Lightning model."""

    def __init__(self):
        super().__init__()
        self.name = "sdxl-lightning"
        self.model_name = SDXL_LIGHTNING_MODEL_URL

    def generate_image(self, ctx):
        prompt = ctx.params.get("prompt", "None provided")
        negative_prompt = ctx.params.get(
            "negative_prompt", SDXL_LIGHTNING_NEGATIVE_PROMPT_DEFAULT
        )
        inference_steps = ctx.params.get(
            "inference_steps", SDXL_LIGHTNING_NUM_INFERENCE_STEPS_DEFAULT
        )
        scheduler = ctx.params.get(
            "scheduler_choices", SDXL_LIGHTNING_SCHEDULER_CHOICES[0]
        )
        guidance_scale = ctx.params.get(
            "guidance_scale", SDXL_LIGHTNING_GUIDANCE_SCALE_DEFAULT
        )
        width = ctx.params.get("width", SDXL_LIGHTNING_SIZE_CHOICES[0])
        height = ctx.params.get("height", SDXL_LIGHTNING_SIZE_CHOICES[0])

        _inputs = {
            "prompt": prompt,
            "num_inference_steps": inference_steps,
            "scheduler": scheduler,
            "guidance_scale": guidance_scale,
            "negative_prompt": negative_prompt,
            "width": int(width),
            "height": int(height),
        }

        response = replicate.run(
            self.model_name,
            input=_inputs,
        )
        if type(response) == list:
            response = response[0]
        return response


class SSD1B(Text2Image):
    """Wrapper for a SSD-1B model."""

    def __init__(self):
        super().__init__()
        self.name = "ssd-1b"
        self.model_name = SSD1B_MODEL_URL

    def generate_image(self, ctx):
        prompt = ctx.params.get("prompt", "None provided")
        inference_steps = ctx.params.get("inference_steps", 25.0)
        scheduler = ctx.params.get("scheduler_choices", "None provided")
        guidance_scale = ctx.params.get("guidance_scale", 7.5)
        negative_prompt = ctx.params.get("negative_prompt", None)

        _inputs = {
            "prompt": prompt,
            "inference_steps": inference_steps,
            "scheduler": scheduler,
            "guidance_scale": guidance_scale,
        }
        if negative_prompt is not None:
            _inputs["negative_prompt"] = negative_prompt

        response = replicate.run(
            self.model_name,
            input=_inputs,
        )
        if type(response) == list:
            response = response[0]
        return response


class Kandinsky(Text2Image):
    """Wrapper for a Kandinsky model."""

    def __init__(self):
        super().__init__()
        self.name = "kandinsky"
        self.model_name = KANDINSKY_MODEL_URL

    def generate_image(self, ctx):
        prompt = ctx.params.get("prompt", "None provided")
        negative_prompt = ctx.params.get("negative_prompt", None)
        width = int(ctx.params.get("width", 512))
        height = int(ctx.params.get("height", 512))
        inference_steps = ctx.params.get("inference_steps", 75)
        inference_steps_prior = ctx.params.get("inference_steps_prior", 25)

        input = {
            "prompt": prompt,
            "width": width,
            "height": height,
            "inference_steps": inference_steps,
            "inference_steps_prior": inference_steps_prior,
        }

        if negative_prompt is not None:
            input["negative_prompt"] = negative_prompt

        response = replicate.run(
            self.model_name,
            input=input,
        )
        if type(response) == list:
            response = response[0]
        return response


class PlaygroundV2(Text2Image):
    """Wrapper for Aesthetic Playground V2 model."""

    def __init__(self):
        super().__init__()
        self.name = "playground-v2"
        self.model_name = PLAYGROUND_V2_MODEL_URL

    def generate_image(self, ctx):
        prompt = ctx.params.get("prompt", "None provided")
        negative_prompt = ctx.params.get("negative_prompt", None)
        width = int(ctx.params.get("width", 1024))
        height = int(ctx.params.get("height", 1024))
        inference_steps = ctx.params.get("inference_steps", 50)
        guidance_scale = ctx.params.get("guide_scale", 3.0)
        scheduler = ctx.params.get("scheduler_choices", "None provided")

        input = {
            "prompt": prompt,
            "width": width,
            "height": height,
            "num_inference_steps": inference_steps,
            "guidance_scale": guidance_scale,
            "scheduler": scheduler,
        }

        if negative_prompt is not None:
            input["negative_prompt"] = negative_prompt

        response = replicate.run(
            self.model_name,
            input=input,
        )
        if type(response) == list:
            response = response[0]
        return response


class LatentConsistencyModel(Text2Image):
    """Wrapper for a Latent Consistency model."""

    def __init__(self):
        super().__init__()
        self.name = "latent-consistency"
        self.model_name = LC_MODEL_URL

    def generate_image(self, ctx):
        prompt = ctx.params.get("prompt", "None provided")
        width = int(ctx.params.get("width", 512))
        height = int(ctx.params.get("height", 512))
        num_inf_steps = int(ctx.params.get("num_inference_steps", 4))
        guide_scale = float(ctx.params.get("guidance_scale", 7.5))
        lcm_origin_steps = int(ctx.params.get("lcm_origin_steps", 50))
        distro = ctx.params.get("model_distribution", "None provided")

        if distro == "replicate":
            response = replicate.run(
                self.model_name,
                input={
                    "prompt": prompt,
                    "width": width,
                    "height": height,
                    "num_inference_steps": num_inf_steps,
                    "guidance_scale": guide_scale,
                    "lcm_origin_steps": lcm_origin_steps,
                },
            )

            if type(response) == list:
                response = response[0]
            return response
        else:
            with add_sys_path(os.path.dirname(os.path.abspath(__file__))):
                # pylint: disable=no-name-in-module,import-error
                from local_t2i_models import lcm

            response = lcm(
                prompt,
                width,
                height,
                num_inf_steps,
                guide_scale,
                lcm_origin_steps,
            )
            return response


class DALLE2(Text2Image):
    """Wrapper for a DALL-E 2 model."""

    def __init__(self):
        super().__init__()
        self.name = "dalle2"

    def generate_image(self, ctx):
        prompt = ctx.params.get("prompt", "None provided")
        size = ctx.params.get("size_choices", "None provided")

        response = openai.OpenAI().images.generate(
            model="dall-e-2", prompt=prompt, n=1, size=size
        )
        return response.data[0].url


class DALLE3(Text2Image):
    """Wrapper for a DALL-E 3 model."""

    def __init__(self):
        super().__init__()
        self.name = "dalle3"

    def generate_image(self, ctx):
        prompt = ctx.params.get("prompt", "None provided")
        size = ctx.params.get("size_choices", "None provided")
        quality = ctx.params.get("quality_choices", "None provided")

        response = openai.OpenAI().images.generate(
            model="dall-e-3", prompt=prompt, n=1, quality=quality, size=size
        )

        revised_prompt = response.data[0].revised_prompt
        ctx.params["revised_prompt"] = revised_prompt

        return response.data[0].url


class VQGANCLIP(Text2Image):
    """Wrapper for a VQGAN-CLIP model."""

    def __init__(self):
        super().__init__()
        self.name = "vqgan-clip"
        self.model_name = VQGAN_MODEL_URL

    def generate_image(self, ctx):
        prompt = ctx.params.get("prompt", "None provided")
        response = replicate.run(self.model_name, input={"prompt": prompt})
        if type(response) == list:
            response = response[0]
        return response


def get_model(model_name):
    mapping = {
        "sd": StableDiffusion,
        "sdxl": SDXL,
        "sdxl-lightning": SDXLLightning,
        "ssd-1b": SSD1B,
        "latent-consistency": LatentConsistencyModel,
        "kandinsky-2.2": Kandinsky,
        "playground-v2": PlaygroundV2,
        "dalle2": DALLE2,
        "dalle3": DALLE3,
        "vqgan-clip": VQGANCLIP,
    }
    return mapping[model_name]()


def set_stable_diffusion_config(sample, ctx):
    sample["stable_diffusion_config"] = fo.DynamicEmbeddedDocument(
        inference_steps=ctx.params.get("inference_steps", "None provided"),
        scheduler=ctx.params.get("scheduler_choices", "None provided"),
        width=ctx.params.get("width_choices", "None provided"),
        height=ctx.params.get("height_choices", "None provided"),
    )


def set_sdxl_config(sample, ctx):
    sample["sdxl_config"] = fo.DynamicEmbeddedDocument(
        inference_steps=ctx.params.get("inference_steps", "None provided"),
        scheduler=ctx.params.get("scheduler_choices", "None provided"),
        guidance_scale=ctx.params.get("guidance_scale", 7.5),
        refiner=SDXL_REFINE_MAP[
            ctx.params.get("refine_choices", "None provided")
        ],
        refine_steps=ctx.params.get("refine_steps", None),
        negative_prompt=ctx.params.get("negative_prompt", None),
        high_noise_frac=ctx.params.get("high_noise_frac", None),
    )


def set_sdxl_lightning_config(sample, ctx):
    sample["sdxl_lightning_config"] = fo.DynamicEmbeddedDocument(
        inference_steps=ctx.params.get(
            "inference_steps", SDXL_LIGHTNING_NUM_INFERENCE_STEPS_DEFAULT
        ),
        scheduler=ctx.params.get("scheduler_choices", "None provided"),
        guidance_scale=ctx.params.get(
            "guidance_scale", SDXL_LIGHTNING_GUIDANCE_SCALE_DEFAULT
        ),
        width=ctx.params.get("width", 1024),
        height=ctx.params.get("height", 1024),
        negative_prompt=ctx.params.get(
            "negative_prompt", SDXL_LIGHTNING_NEGATIVE_PROMPT_DEFAULT
        ),
    )


def set_ssd1b_config(sample, ctx):
    sample["ssd1b_config"] = fo.DynamicEmbeddedDocument(
        inference_steps=ctx.params.get("inference_steps", "None provided"),
        scheduler=ctx.params.get("scheduler_choices", "None provided"),
        guidance_scale=ctx.params.get("guidance_scale", 7.5),
        negative_prompt=ctx.params.get("negative_prompt", None),
    )


def set_kandinsky_config(sample, ctx):
    sample["kandinsky_config"] = fo.DynamicEmbeddedDocument(
        inference_steps=ctx.params.get("inference_steps", 75),
        inference_steps_prior=ctx.params.get("inference_steps_prior", 25),
        width=ctx.params.get("width", 512),
        height=ctx.params.get("height", 512),
        negative_prompt=ctx.params.get("negative_prompt", None),
    )


def set_playground_v2_config(sample, ctx):
    sample["playground_v2_config"] = fo.DynamicEmbeddedDocument(
        inference_steps=ctx.params.get("inference_steps", 50),
        width=ctx.params.get("width", 1024),
        height=ctx.params.get("height", 1024),
        guidance_scale=ctx.params.get("guidance_scale", 3.0),
        scheduler=ctx.params.get("scheduler_choices", "None provided"),
        negative_prompt=ctx.params.get("negative_prompt", None),
    )


def set_latent_consistency_config(sample, ctx):
    sample["latent_consistency_config"] = fo.DynamicEmbeddedDocument(
        inference_steps=ctx.params.get("num_inference_steps", 4),
        guidance_scale=ctx.params.get("guidance_scale", 7.5),
        lcm_origin_steps=ctx.params.get("lcm_origin_steps", 50),
        width=ctx.params.get("width", 512),
        height=ctx.params.get("height", 512),
    )


def set_vqgan_clip_config(sample, ctx):
    return


def set_dalle2_config(sample, ctx):
    sample["dalle2_config"] = fo.DynamicEmbeddedDocument(
        size=ctx.params.get("size_choices", "None provided")
    )


def set_dalle3_config(sample, ctx):
    sample["dalle3_config"] = fo.DynamicEmbeddedDocument(
        size=ctx.params.get("size_choices", "None provided"),
        quality=ctx.params.get("quality_choices", "None provided"),
        revised_prompt=ctx.params.get("revised_prompt", "None provided"),
    )


def set_config(sample, ctx, model_name):
    mapping = {
        "sd": set_stable_diffusion_config,
        "sdxl": set_sdxl_config,
        "sdxl-lightning": set_sdxl_lightning_config,
        "ssd-1b": set_ssd1b_config,
        "latent-consistency": set_latent_consistency_config,
        "kandinsky-2.2": set_kandinsky_config,
        "playground-v2": set_playground_v2_config,
        "dalle2": set_dalle2_config,
        "dalle3": set_dalle3_config,
        "vqgan-clip": set_vqgan_clip_config,
    }

    config_setter = mapping[model_name]
    config_setter(sample, ctx)


def generate_filepath(ctx):
    download_dir = ctx.params.get("download_dir", {})
    if type(download_dir) == dict:
        download_dir = download_dir.get("absolute_path", "/tmp")

    filename = str(uuid.uuid4())[:13].replace("-", "") + ".png"
    return os.path.join(download_dir, filename)


#### MODEL CHOICES ####
def _add_replicate_choices(model_choices):
    model_choices.add_choice("sd", label="Stable Diffusion")
    model_choices.add_choice("sdxl", label="SDXL")
    model_choices.add_choice("sdxl-lightning", label="SDXL Lightning")
    model_choices.add_choice("ssd-1b", label="SSD-1B")
    model_choices.add_choice("kandinsky-2.2", label="Kandinsky 2.2")
    model_choices.add_choice("playground-v2", label="Playground V2")
    if "latent-consistency" not in model_choices.values():
        model_choices.add_choice(
            "latent-consistency", label="Latent Consistency"
        )
    model_choices.add_choice("vqgan-clip", label="VQGAN-CLIP")


def _add_openai_choices(model_choices):
    model_choices.add_choice("dalle2", label="DALL-E2")
    model_choices.add_choice("dalle3", label="DALL-E3")


def _add_diffusers_choices(model_choices):
    if "latent-consistency" not in model_choices.values():
        model_choices.add_choice(
            "latent-consistency", label="Latent Consistency"
        )


#### STABLE DIFFUSION INPUTS ####
def _handle_stable_diffusion_input(ctx, inputs):
    size_choices = SD_SIZE_CHOICES
    width_choices = types.Dropdown(label="Width")
    for size in size_choices:
        width_choices.add_choice(size, label=size)

    inputs.enum(
        "width_choices",
        width_choices.values(),
        default="512",
        view=width_choices,
    )

    height_choices = types.Dropdown(label="Height")
    for size in size_choices:
        height_choices.add_choice(size, label=size)

    inputs.enum(
        "height_choices",
        height_choices.values(),
        default="512",
        view=height_choices,
    )

    inference_steps_slider = types.SliderView(
        label="Num Inference Steps",
        componentsProps={"slider": {"min": 1, "max": 500, "step": 1}},
    )
    inputs.int("inference_steps", default=50, view=inference_steps_slider)

    scheduler_choices_dropdown = types.Dropdown(label="Scheduler")
    for scheduler in SD_SCHEDULER_CHOICES:
        scheduler_choices_dropdown.add_choice(scheduler, label=scheduler)

    inputs.enum(
        "scheduler_choices",
        scheduler_choices_dropdown.values(),
        default="K_EULER",
        view=scheduler_choices_dropdown,
    )


#### SDXL INPUTS ####
def _handle_sdxl_input(ctx, inputs):

    inputs.str("negative_prompt", label="Negative Prompt", required=False)

    scheduler_choices_dropdown = types.Dropdown(label="Scheduler")
    for scheduler in SDXL_SCHEDULER_CHOICES:
        scheduler_choices_dropdown.add_choice(scheduler, label=scheduler)

    inputs.enum(
        "scheduler_choices",
        scheduler_choices_dropdown.values(),
        default="K_EULER",
        view=scheduler_choices_dropdown,
    )

    inference_steps_slider = types.SliderView(
        label="Num Inference Steps",
        componentsProps={"slider": {"min": 1, "max": 100, "step": 1}},
    )
    inputs.int("inference_steps", default=50, view=inference_steps_slider)

    guidance_scale_slider = types.SliderView(
        label="Guidance Scale",
        componentsProps={"slider": {"min": 0.0, "max": 10.0, "step": 0.1}},
    )
    inputs.float("guidance_scale", default=7.5, view=guidance_scale_slider)

    refiner_choices_dropdown = types.Dropdown(
        label="Refiner",
        description="Which refine style to use",
    )
    for refiner in SDXL_REFINE_CHOICES:
        refiner_choices_dropdown.add_choice(refiner, label=refiner)

    inputs.enum(
        "refine_choices",
        refiner_choices_dropdown.values(),
        default="None",
        view=refiner_choices_dropdown,
    )

    rfc = SDXL_REFINE_MAP[ctx.params.get("refine_choices", "None")]
    if rfc == "base_image_refiner":
        _default = ctx.params.get("inference_steps", 50)
        refine_steps_slider = types.SliderView(
            label="Num Refine Steps",
            componentsProps={"slider": {"min": 1, "max": _default, "step": 1}},
        )
        inputs.int(
            "refine_steps",
            label="Refine Steps",
            description="The number of steps to refine",
            default=_default,
            view=refine_steps_slider,
        )
    elif rfc == "expert_ensemble_refiner":
        inputs.float(
            "high_noise_frac",
            label="High Noise Fraction",
            description="The fraction of noise to use",
            default=0.8,
        )


#### SDXL LIGHTNING INPUTS ####
def _handle_sdxl_lightning_input(ctx, inputs):
    size_choices = SDXL_LIGHTNING_SIZE_CHOICES
    width_choices = types.Dropdown(label="Width")
    for size in size_choices:
        width_choices.add_choice(size, label=size)

    inputs.enum(
        "width",
        width_choices.values(),
        default=SDXL_LIGHTNING_SIZE_CHOICES[0],
        view=width_choices,
    )

    height_choices = types.Dropdown(label="Height")
    for size in size_choices:
        height_choices.add_choice(size, label=size)

    inputs.enum(
        "height",
        height_choices.values(),
        default=SDXL_LIGHTNING_SIZE_CHOICES[0],
        view=height_choices,
    )

    inputs.str(
        "negative_prompt",
        label="Negative Prompt",
        required=False,
        default=SDXL_LIGHTNING_NEGATIVE_PROMPT_DEFAULT,
    )

    inference_steps_slider = types.SliderView(
        label="Num Inference Steps",
        componentsProps={"slider": {"min": 1, "max": 10, "step": 1}},
    )
    inputs.int(
        "inference_steps",
        default=SDXL_LIGHTNING_NUM_INFERENCE_STEPS_DEFAULT,
        view=inference_steps_slider,
    )

    guidance_scale_slider = types.SliderView(
        label="Guidance Scale",
        componentsProps={"slider": {"min": 0.0, "max": 10.0, "step": 0.1}},
    )
    inputs.float(
        "guidance_scale",
        default=SDXL_LIGHTNING_GUIDANCE_SCALE_DEFAULT,
        view=guidance_scale_slider,
    )

    scheduler_choices_dropdown = types.Dropdown(label="Scheduler")
    for scheduler in SDXL_LIGHTNING_SCHEDULER_CHOICES:
        scheduler_choices_dropdown.add_choice(scheduler, label=scheduler)

    inputs.enum(
        "scheduler_choices",
        scheduler_choices_dropdown.values(),
        default="K_EULER",
        view=scheduler_choices_dropdown,
    )


#### SSD-1B INPUTS ####
def _handle_ssd1b_input(ctx, inputs):
    inputs.int("width", label="Width", default=768)
    inputs.int("height", label="Height", default=768)
    inputs.str("negative_prompt", label="Negative Prompt", required=False)

    scheduler_choices_dropdown = types.Dropdown(label="Scheduler")
    for scheduler in SSD1B_SCHEDULER_CHOICES:
        scheduler_choices_dropdown.add_choice(scheduler, label=scheduler)

    inputs.enum(
        "scheduler_choices",
        scheduler_choices_dropdown.values(),
        default="K_EULER",
        view=scheduler_choices_dropdown,
    )

    inference_steps_slider = types.SliderView(
        label="Num Inference Steps",
        componentsProps={"slider": {"min": 1, "max": 100, "step": 1}},
    )
    inputs.int("inference_steps", default=25, view=inference_steps_slider)

    guidance_scale_slider = types.SliderView(
        label="Guidance Scale",
        componentsProps={"slider": {"min": 0.0, "max": 10.0, "step": 0.1}},
    )
    inputs.float("guidance_scale", default=7.5, view=guidance_scale_slider)


#### KANDINSKY INPUTS ####
def _handle_kandinsky_input(ctx, inputs):

    size_choices = KANDINSDKY_SIZE_CHOICES
    width_choices = types.Dropdown(label="Width")
    for size in size_choices:
        width_choices.add_choice(size, label=size)

    inputs.enum(
        "width",
        width_choices.values(),
        default="512",
        view=width_choices,
    )

    height_choices = types.Dropdown(label="Height")
    for size in size_choices:
        height_choices.add_choice(size, label=size)

    inputs.enum(
        "height",
        height_choices.values(),
        default="512",
        view=height_choices,
    )
    inputs.str("negative_prompt", label="Negative Prompt", required=False)

    inference_steps_slider = types.SliderView(
        label="Num Inference Steps",
        componentsProps={"slider": {"min": 1, "max": 100, "step": 1}},
    )
    inputs.int("inference_steps", default=75, view=inference_steps_slider)

    inference_steps_prior_slider = types.SliderView(
        label="Num Inference Steps Prior",
        componentsProps={"slider": {"min": 1, "max": 50, "step": 1}},
    )
    inputs.int(
        "inference_steps_prior",
        default=25,
        view=inference_steps_prior_slider,
    )


#### PLAYGROUND V2 INPUTS ####
def _handle_playground_v2_input(ctx, inputs):
    inputs.int("width", label="Width", default=1024)
    inputs.int("height", label="Height", default=1024)

    inputs.str("negative_prompt", label="Negative Prompt", required=False)

    inference_steps_slider = types.SliderView(
        label="Num Inference Steps",
        componentsProps={"slider": {"min": 1, "max": 100, "step": 1}},
    )
    inputs.int("inference_steps", default=50, view=inference_steps_slider)

    guidance_scale_slider = types.SliderView(
        label="Guidance Scale",
        componentsProps={"slider": {"min": 0.0, "max": 10.0, "step": 0.1}},
    )
    inputs.float("guidance_scale", default=3.0, view=guidance_scale_slider)

    scheduler_choices_dropdown = types.Dropdown(label="Scheduler")
    for scheduler in PLAYGROUND_V2_SCHEDULER_CHOICES:
        scheduler_choices_dropdown.add_choice(scheduler, label=scheduler)

    inputs.enum(
        "scheduler_choices",
        scheduler_choices_dropdown.values(),
        default="K_EULER_ANCESTRAL",
        view=scheduler_choices_dropdown,
    )


#### LATENT CONSISTENCY INPUTS ####
def _handle_latent_consistency_input(ctx, inputs):

    replicate_flag = allows_replicate_models()
    diffusers_flag = allows_diffusers_models()

    if not replicate_flag:
        ctx.params["model_distribution"] = "diffusers"
    elif not diffusers_flag:
        ctx.params["model_distribution"] = "replicate"
    else:
        model_distribution_choices = types.Dropdown(label="Model Distribution")
        model_distribution_choices.add_choice("diffusers", label="Diffusers")
        model_distribution_choices.add_choice("replicate", label="Replicate")
        inputs.enum(
            "model_distribution",
            model_distribution_choices.values(),
            default="diffusers",
            view=model_distribution_choices,
        )

    inputs.int("width", label="Width", default=512)
    inputs.int("height", label="Height", default=512)

    inference_steps_slider = types.SliderView(
        label="Num Inference Steps",
        componentsProps={"slider": {"min": 1, "max": 50, "step": 1}},
    )
    inputs.int("num_inference_steps", default=4, view=inference_steps_slider)

    lcm_origin_steps_slider = types.SliderView(
        label="LCM Origin Steps",
        componentsProps={"slider": {"min": 1, "max": 100, "step": 1}},
    )
    inputs.int("lcm_origin_steps", default=50, view=lcm_origin_steps_slider)

    guidance_scale_slider = types.SliderView(
        label="Guidance Scale",
        componentsProps={"slider": {"min": 0.0, "max": 10.0, "step": 0.1}},
    )
    inputs.float("guidance_scale", default=7.5, view=guidance_scale_slider)


#### DALLE2 INPUTS ####
def _handle_dalle2_input(ctx, inputs):
    size_choices_dropdown = types.Dropdown(label="Size")
    for size in DALLE2_SIZE_CHOICES:
        size_choices_dropdown.add_choice(size, label=size)

    inputs.enum(
        "size_choices",
        size_choices_dropdown.values(),
        default="512x512",
        view=size_choices_dropdown,
    )


#### DALLE3 INPUTS ####
def _handle_dalle3_input(ctx, inputs):
    size_choices_dropdown = types.Dropdown(label="Size")
    for size in DALLE3_SIZE_CHOICES:
        size_choices_dropdown.add_choice(size, label=size)

    inputs.enum(
        "size_choices",
        size_choices_dropdown.values(),
        default="1024x1024",
        view=size_choices_dropdown,
    )

    quality_choices_dropdown = types.Dropdown(label="Quality")
    for quality in DALLE3_QUALITY_CHOICES:
        quality_choices_dropdown.add_choice(quality, label=quality)

    inputs.enum(
        "quality_choices",
        quality_choices_dropdown.values(),
        default="standard",
        view=quality_choices_dropdown,
    )


#### VQGAN-CLIP INPUTS ####
def _handle_vqgan_clip_input(ctx, inputs):
    return


INPUT_MAPPER = {
    "sd": _handle_stable_diffusion_input,
    "sdxl": _handle_sdxl_input,
    "sdxl-lightning": _handle_sdxl_lightning_input,
    "ssd-1b": _handle_ssd1b_input,
    "kandinsky-2.2": _handle_kandinsky_input,
    "playground-v2": _handle_playground_v2_input,
    "latent-consistency": _handle_latent_consistency_input,
    "dalle2": _handle_dalle2_input,
    "dalle3": _handle_dalle3_input,
    "vqgan-clip": _handle_vqgan_clip_input,
}


def _handle_input(ctx, inputs):
    model_name = ctx.params.get("model_choices", "sd")
    model_input_handler = INPUT_MAPPER[model_name]
    model_input_handler(ctx, inputs)


def _resolve_download_dir(ctx, inputs):
    if len(ctx.dataset) == 0:
        file_explorer = types.FileExplorerView(
            choose_dir=True,
            button_label="Choose a directory...",
        )
        inputs.file(
            "download_dir",
            required=True,
            description="Choose a location to store downloaded images",
            view=file_explorer,
        )
    else:
        base_dir = os.path.dirname(ctx.dataset.first().filepath).split("/")
        ctx.params["download_dir"] = "/".join(base_dir)


def _handle_calling(uri, sample_collection, prompt, model_name, **kwargs):
    ctx = dict(view=sample_collection.view())
    params = dict(kwargs)
    params["prompt"] = prompt
    params["model_choices"] = model_name

    return foo.execute_operator(uri, ctx, params=params)


class Txt2Image(foo.Operator):
    @property
    def config(self):
        _config = foo.OperatorConfig(
            name="txt2img",
            label="Text to Image: Generate Image from Text",
            dynamic=True,
        )
        _config.icon = "/assets/icon.svg"
        return _config

    def resolve_input(self, ctx):
        inputs = types.Object()
        _resolve_download_dir(ctx, inputs)

        replicate_flag = allows_replicate_models()
        openai_flag = allows_openai_models()
        diffusers_flag = allows_diffusers_models()

        any_flag = replicate_flag or openai_flag or diffusers_flag
        if not any_flag:
            inputs.message(
                "message",
                label="No models available.",
                descriptions=(
                    "You must install one of `replicate`, `openai`, of `diffusers`",
                    " to use this plugin. ",
                ),
            )
            return types.Property(inputs)

        model_choices = types.Dropdown()
        if replicate_flag:
            _add_replicate_choices(model_choices)
        if openai_flag:
            _add_openai_choices(model_choices)
        if diffusers_flag:
            _add_diffusers_choices(model_choices)
        inputs.enum(
            "model_choices",
            model_choices.values(),
            default=model_choices.choices[0].value,
            label="Model",
            description="Choose a model to generate images",
            view=model_choices,
        )

        inputs.str(
            "prompt",
            label="Prompt",
            description="The prompt to generate an image from",
            required=True,
        )
        _handle_input(ctx, inputs)
        return types.Property(inputs)

    def execute(self, ctx):
        model_name = ctx.params.get("model_choices", "None provided")
        model = get_model(model_name)
        prompt = ctx.params.get("prompt", "None provided")

        response = model.generate_image(ctx)
        filepath = generate_filepath(ctx)

        if type(response) == str:
            ## served models return a url
            image_url = response
            download_image(image_url, filepath)
        else:
            ## local models return a PIL image
            response.save(filepath)

        sample = fo.Sample(
            filepath=filepath,
            tags=["generated"],
            model=model.name,
            prompt=prompt,
            date_created=datetime.now(),
        )
        set_config(sample, ctx, model_name)

        dataset = ctx.dataset
        dataset.add_sample(sample, dynamic=True)

        if dataset.get_dynamic_field_schema() is not None:
            dataset.add_dynamic_sample_fields()
            ctx.trigger("reload_dataset")
        else:
            ctx.trigger("reload_samples")

    def list_models(self):
        return list(INPUT_MAPPER.keys())

    def __call__(self, sample_collection, prompt, model_name, **kwargs):
        _handle_calling(
            self.uri, sample_collection, prompt, model_name, **kwargs
        )


def register(plugin):
    plugin.register(Txt2Image)
