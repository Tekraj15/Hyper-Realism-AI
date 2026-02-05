import torch
from diffusers import FluxPipeline, FluxTransformer2DModel
from .base_engine import GenerativeEngine

class AppleSiliconEngine(GenerativeEngine):
    def __init__(self, config):
        self.config = config
        self.pipe = None
        self.device = "mps" if torch.backends.mps.is_available() else "cpu"
        print(f"   [AppleSiliconEngine] Initializing with GGUF Quantization ({self.device})...")
        
    def load_models(self):
        if self.pipe is None:
            print(f"   [AppleSiliconEngine] Loading model: FLUX.1-dev (Q8_0 GGUF)")
            
            # Load the compressed GGUF transformer (~6GB instead of ~24GB)
            transformer = FluxTransformer2DModel.from_single_file(
                "https://huggingface.co/city96/FLUX.1-dev-gguf/blob/main/flux1-dev-Q8_0.gguf",
                torch_dtype=torch.bfloat16
            )
        
            # Load the rest of the pipeline with the GGUF transformer
            self.pipe = FluxPipeline.from_pretrained(
                "black-forest-labs/FLUX.1-dev",
                transformer=transformer,
                torch_dtype=torch.bfloat16
            )
            
            # Critical for Mac - offloads components to CPU when not in use
            self.pipe.enable_model_cpu_offload()
            
            print("   [AppleSiliconEngine] Model loaded successfully.")

    def generate(self, prompt, **kwargs):
        self.load_models()
        
        # Extract parameters with defaults from config or kwargs
        width = kwargs.get('width', self.config['generation']['defaults']['width'])
        height = kwargs.get('height', self.config['generation']['defaults']['height'])
        num_steps = kwargs.get('steps', self.config['generation']['defaults']['steps'])
        guidance_scale = kwargs.get('guidance', self.config['generation']['defaults']['guidance'])
        seed = kwargs.get('seed', None)
        
        # Prepend trigger word if not present
        trigger_word = self.config['models']['dev']['trigger_word']
        if trigger_word not in prompt:
            prompt = f"{trigger_word} {prompt}"
            
        print(f"   [AppleSiliconEngine] Generating image for prompt: '{prompt[:50]}...'")
        
        # Set up generator for reproducibility
        generator = None
        if seed is not None and seed > 0:
            generator = torch.Generator(device="cpu").manual_seed(int(seed))
        
        # Generate image using the GGUF pipeline
        result = self.pipe(
            prompt=prompt,
            width=width,
            height=height,
            num_inference_steps=num_steps,
            guidance_scale=guidance_scale,
            generator=generator
        )
        
        return result.images[0]