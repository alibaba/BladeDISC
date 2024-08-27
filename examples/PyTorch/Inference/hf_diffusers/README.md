# Accelerate Inference of Stable Diffusion using BladeDISC

*(under development)*
BladeDISC can compile PyTorch models in Stable Diffusion pipeline to improve the inference speed.
A general workflow is like: export model and call BladeDISC to compile, then wrap optimized model
into original pipeline.  
To further simplify the optimization workflow, we provide a adapter for Huggingface Diffusers library.

## Usage

### Use Pipeline Adapter

```python
from blade_adapter import BladeStableDiffusionPipeline

# use adapter to load pipe and optimize models:
pipe = BladeStableDiffusionPipeline.from_pretrained('runwayml/stable-diffusion-v1-5')

# use optimized pipeline like original one:
prompt = "a photo of an astronaut riding a horse on mars"
image = pipe(prompt).images[0]

# save and load optimized pipeline (to avoid run compilation from original models every time):
pipe.saved_pretrained('cached/dir/stable-diffusion-v1-5-blade-opt')
pipe = BladeStableDiffusionPipeline.from_pretrained('cached/dir/stable-diffusion-v1-5-blade-opt')

```


### Use Model Adapter

*(TBD)*