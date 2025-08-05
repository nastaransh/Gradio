import numpy as np
import gradio as gr

def flip_text(x):

    return x[::-1], f"Character Count: {len(x)}"

def flip_image(x):
    if x is None:
        raise gr.Error("⚠️ Please upload an image before flipping.")
    return x, np.fliplr(x)

with gr.Blocks(title="🔁 Flip Text and Image Demo") as demo:
    gr.Markdown("## 🔄 Flip Text or Image\nUse tabs below to flip text or images interactively.")

    with gr.Tab("📝 Flip Text"):
        gr.Markdown("### Enter your text below:")
        with gr.Row():
            text_input = gr.Textbox(lines=3, label="Input Text")
            text_output = gr.Textbox(lines=3, label="Flipped Text")
        with gr.Row():
            char_count = gr.Textbox(label="Character Info", interactive=False)
        with gr.Row():
            text_button = gr.Button("🔁 Flip")
            text_clear = gr.Button("❌ Clear")

        text_button.click(flip_text, inputs=text_input, outputs=[text_output, char_count])
        text_clear.click(lambda: ("", "", ""), outputs=[text_input, text_output, char_count])

    with gr.Tab("🖼️ Flip Image"):
        gr.Markdown("### Upload an image to flip it horizontally.")
        with gr.Row():
            image_input = gr.Image(label="Original")
            image_output = gr.Image(label="Flipped")
        with gr.Row():
            image_button = gr.Button("🔁 Flip", interactive=False)  # Disabled initially
            image_clear = gr.Button("❌ Clear")

        # Enable Flip button only if image is uploaded
        def toggle_button(img):
            return gr.update(interactive=img is not None)

        image_input.change(toggle_button, inputs=image_input, outputs=image_button)

        # Flip image on button click
        image_button.click(flip_image, inputs=image_input, outputs=[image_input, image_output])

        # Clear images and disable flip button again
        def clear_images():
            return None, None, gr.update(interactive=False)

        image_clear.click(clear_images, outputs=[image_input, image_output, image_button])

    with gr.Accordion("📌 More Settings", open=False):
        gr.Markdown("This slider currently does nothing — but could control intensity in a real app.")
        temp_slider = gr.Slider(0, 1, value=0.1, step=0.1, interactive=True, label="(Demo Slider)")

demo.launch()
