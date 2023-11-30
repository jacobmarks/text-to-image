from diffusers import DiffusionPipeline


def get_cache():
    g = globals()
    if "_local_t2i_models" not in g:
        g["_local_t2i_models"] = {}

    return g["_local_t2i_models"]


def lcm(
    prompt,
    width,
    height,
    num_inference_steps,
    guide_scale,
    lcm_origin_steps,
):
    if "lcm" not in get_cache():
        get_cache()["lcm"] = DiffusionPipeline.from_pretrained(
            "SimianLuo/LCM_Dreamshaper_v7"
        )

    pipe = get_cache()["lcm"]
    images = pipe(
        prompt=prompt,
        num_inference_steps=num_inference_steps,
        guidance_scale=guide_scale,
        lcm_origin_steps=lcm_origin_steps,
        output_type="pil",
        width=width,
        height=height,
    ).images
    return images[0]
