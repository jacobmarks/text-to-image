## AI Art Gallery Plugin

![ssd1b](https://github.com/jacobmarks/ai-art-gallery/assets/12500356/f5202d68-c5c1-44c7-b662-98d98e5c05aa)

### Updates

- **2021-10-30**: Added support for Segmind Stable Diffusion (SSD-1B) Model
- **2021-10-23**: Added support for Latent Consistency Model
- **2023-10-18**: Added support for SDXL, operator icon, and download location selection

### Plugin Overview

This plugin is a Python plugin that allows you to generate images from text
prompts and add them directly into your dataset.

It demonstrates how to do the following:

- use Python to create an operator with different options depending on user
  choices
- use `componentProps` to customize the UI
- download images from URL and add them to the dataset

:warning: This plugin is only verified to work for local datasets. It may not
work for remote datasets.

### Supported Models

This version of the plugin supports the following models:

- [SDXL](https://replicate.com/stability-ai/sdxl)
- [Stable Diffusion](https://replicate.com/stability-ai/stable-diffusion)
- [Segmind Stable Diffusion (SSD-1B)](https://replicate.com/lucataco/ssd-1b/)
- [Latent Consistency Model](https://replicate.com/luosiallen/latent-consistency-model/)
- [DALL-E2](https://openai.com/dall-e-2)
- [VQGAN-CLIP](https://replicate.com/mehdidc/feed_forward_vqgan_clip)

It is straightforward to add support for other models!

## Installation

```shell
fiftyone plugins download https://github.com/jacobmarks/ai-art-gallery
```

If you want to use Replicate models, you will
need to `pip install replicate` and set the environment variable
`REPLICATE_API_TOKEN` with your API token.

If you want to use DALL-E2, you will need to `pip install openai` and set the
environment variable `OPENAI_API_KEY` with your API key.

Refer to the [main README](https://github.com/voxel51/fiftyone-plugins) for
more information about managing downloaded plugins and developing plugins
locally.

## Operators

### `txt2img`

- Generates an image from a text prompt and adds it to the dataset
