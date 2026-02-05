import gradio as gr
import time
import os
import uuid
import zipfile
from PIL import Image

def create_ui(engine, config):
    """
    Constructs the Gradio Blocks UI and binds it to the provided engine.
    """
    
    # Extract style data from config
    styles = config.get('styles', {})
    style_names = list(styles.keys())
    default_style = "Style Zero"
    
    # Theme & CSS for a premium look
    theme = gr.themes.Soft(
        primary_hue="blue",
        secondary_hue="slate",
        neutral_hue="slate",
    ).set(
        button_primary_background_fill="*primary_500",
        button_primary_background_fill_hover="*primary_600",
    )

    css = """
    #header { text-align: center; margin-bottom: 20px; }
    #header h1 { font-weight: 800; font-size: 2.5em; letter-spacing: -1px; }
    #generate-btn { font-weight: 600; height: 50px; }
    .gradio-container { max-width: 1200px !important; }
    """

    with gr.Blocks(theme=theme, css=css) as demo:
        with gr.Column(elem_id="header"):
            gr.Markdown("# 🌀 Hyper-Realism-AI")
            gr.Markdown("### Frontier-class Image Synthesis for Consumers")

        with gr.Row():
            with gr.Column(scale=1):
                prompt = gr.Textbox(
                    label="Prompt",
                    placeholder="Describe the image you want to generate...",
                    lines=3
                )
                
                with gr.Row():
                    style_choice = gr.Dropdown(
                        choices=style_names,
                        value=default_style,
                        label="Quality Preset"
                    )
                    
                with gr.Accordion("Advanced Settings", open=False):
                    negative_prompt = gr.Textbox(
                        label="Negative Prompt",
                        placeholder="Elements to exclude...",
                        lines=2
                    )
                    
                    with gr.Row():
                        width = gr.Slider(512, 1536, value=config['generation']['defaults']['width'], step=64, label="Width")
                        height = gr.Slider(512, 1536, value=config['generation']['defaults']['height'], step=64, label="Height")
                    
                    with gr.Row():
                        guidance_scale = gr.Slider(1.0, 20.0, value=config['generation']['defaults']['guidance'], step=0.5, label="Guidance Scale")
                        num_steps = gr.Slider(1, 50, value=config['generation']['defaults']['steps'], step=1, label="Inference Steps")
                    
                    with gr.Row():
                        seed = gr.Number(label="Seed (leave -1 for random)", value=-1, precision=0)
                        batch_size = gr.Slider(1, 4, value=1, step=1, label="Batch Size")

                generate_btn = gr.Button("🚀 Generate Images", variant="primary", elem_id="generate-btn")
                
            with gr.Column(scale=1):
                output_gallery = gr.Gallery(
                    label="Generated Images", 
                    show_label=False, 
                    elem_id="gallery",
                    columns=[2], 
                    rows=[2], 
                    object_fit="contain", 
                    height="600px"
                )
                
                with gr.Row():
                    download_btn = gr.File(label="Download Batch (ZIP)", visible=False)
                
                generation_info = gr.Markdown("")

        def run_generation(prompt, style, neg_prompt, w, h, guidance, steps, seed_val, batch):
            start_time = time.time()
            
            # 1. Apply Style Preset
            style_data = styles.get(style, styles.get(default_style))
            final_prompt = style_data['prompt'].format(prompt=prompt)
            
            # 2. Handle Seed
            seeds = []
            if seed_val == -1:
                import random
                for _ in range(batch):
                    seeds.append(random.randint(0, config['generation']['defaults']['seed_max']))
            else:
                for i in range(batch):
                    seeds.append(seed_val + i)
            
            # 3. Generate
            results = []
            output_dir = "outputs"
            os.makedirs(output_dir, exist_ok=True)
            
            batch_id = str(uuid.uuid4())[:8]
            file_paths = []
            
            for i, current_seed in enumerate(seeds):
                print(f"   [UI] Generating batch {i+1}/{batch} with seed {current_seed}...")
                image = engine.generate(
                    prompt=final_prompt,
                    width=w,
                    height=h,
                    guidance=guidance,
                    steps=steps,
                    seed=current_seed,
                    negative_prompt=neg_prompt
                )
                
                # Save image
                filename = f"gen_{batch_id}_{i}.png"
                filepath = os.path.join(output_dir, filename)
                image.save(filepath)
                results.append(image)
                file_paths.append(filepath)
            
            # 4. Create ZIP if batch > 1
            zip_path = None
            if batch > 1:
                zip_filename = f"batch_{batch_id}.zip"
                zip_path = os.path.join(output_dir, zip_filename)
                with zipfile.ZipFile(zip_path, 'w') as zipf:
                    for fp in file_paths:
                        zipf.write(fp, os.path.basename(fp))
            
            total_time = time.time() - start_time
            info_text = f"Generation complete in **{total_time:.2f}s**. Seeds: {seeds}"
            
            return results, info_text, gr.update(value=zip_path, visible=(zip_path is not None))

        generate_btn.click(
            fn=run_generation,
            inputs=[
                prompt, style_choice, negative_prompt, 
                width, height, guidance_scale, num_steps, 
                seed, batch_size
            ],
            outputs=[output_gallery, generation_info, download_btn]
        )
        
    return demo