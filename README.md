## Text-to-Image Plugin

![ssd1b](https://github.com/jacobmarks/ai-art-gallery/assets/12500356/f5202d68-c5c1-44c7-b662-98d98e5c05aa)

### Updates

- **2023-12-19**: Added support for Kandinsky-2.2 and Playground V2 models
- **2023-11-30**: Version 1.2.0
  - adds local model running via `diffusers` (>=0.24.0)
  - adds [calling from the Python SDK](#python-sdk)!
  - :warning: **BREAKING CHANGE**: the plugin and operator URIs have been changed from `ai_art_gallery` to `text_to_image`. If you have any saved pipelines that use the plugin, you will need to update the URIs.
- **2023-11-08**: Version 1.1.0 adds support for DALLE-3 Model — upgrade to `openai>=1.1.0` to use 😄
- **2023-10-30**: Added support for Segmind Stable Diffusion (SSD-1B) Model
- **2023-10-23**: Added support for Latent Consistency Model
- **2023-10-18**: Added support for SDXL, operator icon, and download location selection

### Plugin Overview

This plugin is a Python plugin that allows you to generate images from text
prompts and add them directly into your dataset.

:warning: This plugin is only verified to work for local datasets. It may not
work for remote datasets.

### Supported Models

This version of the plugin supports the following models:

- [DALL-E2](https://openai.com/dall-e-2)
- [DALL-E3](https://openai.com/dall-e-3)
- [Kandinsky-2.2](https://replicate.com/ai-forever/kandinsky-2.2)
- [Latent Consistency Model](https://replicate.com/luosiallen/latent-consistency-model/)
- [Playground V2](https://replicate.com/playgroundai/playground-v2-1024px-aesthetic)
- [SDXL](https://replicate.com/stability-ai/sdxl)
- [SDXL-Lighting](https://replicate.com/lucataco/sdxl-lightning-4step)
- [Segmind Stable Diffusion (SSD-1B)](https://replicate.com/lucataco/ssd-1b/)
- [Stable Diffusion](https://replicate.com/stability-ai/stable-diffusion)
- [VQGAN-CLIP](https://replicate.com/mehdidc/feed_forward_vqgan_clip)

It is straightforward to add support for other models!

## Watch On Youtube

[![Video Thumbnail](https://img.youtube.com/vi/qJNEyC_FqG0/0.jpg)](https://www.youtube.com/watch?v=qJNEyC_FqG0&list=PLuREAXoPgT0RZrUaT0UpX_HzwKkoB-S9j&index=2)

## Installation

```shell
fiftyone plugins download https://github.com/jacobmarks/text-to-image
```

If you want to use Replicate models, you will
need to `pip install replicate` and set the environment variable
`REPLICATE_API_TOKEN` with your API token.

If you want to use DALL-E2 or DALL-E3, you will need to `pip install openai` and set the
environment variable `OPENAI_API_KEY` with your API key.

To run the Latency Consistency model locally with Hugging Face's diffusers library,
you will need `diffusers>=0.24.0`. If you need to, you can install it with
`pip install diffusers>=0.24.0`.

Refer to the [main README](https://github.com/voxel51/fiftyone-plugins) for
more information about managing downloaded plugins and developing plugins
locally.

## Operators

### `txt2img`

- Generates an image from a text prompt and adds it to the dataset

### Python SDK

You can also use the `txt2img` operators from the Python SDK!

⚠️ Due to the way Jupyter Notebooks interact with asyncio, this will not work in a Jupyter Notebook. You will need to run this code in a Python script or in a Python console.

```python
import fiftyone as fo
import fiftyone.operators as foo
import fiftyone.zoo as foz

dataset = fo.load_dataset("quickstart")

## Access the operator via its URI (plugin name + operator name)
t2i = foo.get_operator("@jacobmarks/text_to_image/txt2img")

## Run the operator

prompt = "A dog sitting in a field"
t2i(dataset, prompt=prompt, model_name="latent-consistency", delegate=False)

## Pass in model-specific arguments
t2i(
    dataset,
    prompt=prompt,
    model_name="latent-consistency",
    delegate=False,
    width=768,
    height=768,
    num_inference_steps=8,
)
```
