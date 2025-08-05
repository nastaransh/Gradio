import torch
from pytorch_pretrained_biggan import BigGAN, one_hot_from_names, truncated_noise_sample
import numpy as np
from PIL import Image
import gradio as gr
import nltk

# Ensure WordNet is available
try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')

# Load model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = BigGAN.from_pretrained('biggan-deep-128').to(device)
model.eval()

# Predefined dog breed list
dog_breeds = [
    "golden retriever", "labrador retriever", "german shepherd", "pomeranian",
    "chihuahua", "pug", "dalmatian", "great dane", "siberian husky",
    "doberman", "beagle", "border collie", "bull terrier", "english setter",
    "yorkshire terrier", "boston bull", "miniature poodle"
]

# Tensor → PIL (upscaled)
def convert_to_pil(tensor, upscale=4):
    array = ((tensor.permute(1, 2, 0).cpu().numpy() + 1) * 127.5).astype(np.uint8)
    image = Image.fromarray(array)
    return image.resize((image.width * upscale, image.height * upscale), Image.LANCZOS)

# Generate single image

def generate_image(input_mode, dropdown_value, textbox_value, seed, truncation, scale):
    try:
        # Determine class name
        class_name = dropdown_value if input_mode == "Select from list" else textbox_value.strip()
        if not class_name:
            raise gr.Error("Breed name cannot be empty.")

        # Generate latent noise
        np.random.seed(seed)
        noise = truncated_noise_sample(truncation=truncation, batch_size=1)

        # Generate one-hot vector for class
        class_vec = one_hot_from_names([class_name], batch_size=1)

        # Validate the class vector
        if not isinstance(class_vec, np.ndarray) or not np.any(class_vec):
            raise gr.Error( "Please select a valid breed.")

        # Convert inputs to tensors
        noise_tensor = torch.from_numpy(noise).to(device)
        class_tensor = torch.from_numpy(class_vec).to(device)

        # Generate image
        with torch.no_grad():
            output = model(noise_tensor, class_tensor, truncation)

        # Convert tensor to image
        return convert_to_pil(output[0], upscale=scale)

    except gr.Error as ge:
        # Specific errors shown to user in UI
        raise ge
    except Exception as e:
        # Catch all unexpected errors and show a generic message
        raise gr.Error(f"Something went wrong: {str(e)}")


# Gradio UI
with gr.Blocks() as demo:
    gr.Markdown("# 🐶 High-Quality BigGAN Dog Generator")
    gr.Markdown("Select or type a dog breed to generate a high-quality image using BigGAN.")

    input_mode = gr.Radio(["Select from list", "Type breed manually"], value="Select from list", label="Input Mode")
    
    dropdown = gr.Dropdown(dog_breeds, value="golden retriever", label="Choose Dog Breed", visible=True )
    textbox = gr.Textbox(placeholder="e.g., samoyed, irish terrier", label="Custom Breed Name", visible=False)

    seed_slider = gr.Slider(0, 9999, step=1, value=42, label="Random Seed")
    trunc_slider = gr.Slider(0.1, 1.0, step=0.05, value=0.3, label="Truncation (Lower = Sharper, Less Noise)")
    scale_slider = gr.Slider(1, 5, step=1, value=4, label="Scale Factor (1 = 128px, 4 = 512px)")

    generate_btn = gr.Button("Generate 🐾")
    image_output = gr.Image(type="pil", label="Generated Image")

    # Update visibility based on radio choice
    def toggle_inputs(choice):
        return gr.update(visible=(choice == "Select from list")), gr.update(visible=(choice == "Type breed manually"))

    input_mode.change(toggle_inputs, inputs=input_mode, outputs=[dropdown, textbox])
    
    generate_btn.click(
        fn=generate_image,
        inputs=[input_mode, dropdown, textbox, seed_slider, trunc_slider, scale_slider],
        outputs=image_output
    )

demo.launch()

