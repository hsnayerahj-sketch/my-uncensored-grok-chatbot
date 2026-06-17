import streamlit as st
from diffusers import StableDiffusionImg2ImgPipeline
import torch
from PIL import Image
import io
import time
from groq import Groq
import os

st.set_page_config(page_title="Free Local NSFW img2img", layout="wide")

st.title("🎨 Free Local NSFW img2img Studio")
st.markdown("**100% Free • Unlimited • Runs on your PC • Nudity & Explicit OK**")

# Sidebar
with st.sidebar:
    st.markdown("## Settings")
    device = st.selectbox("Device", ["cpu", "cuda"] if torch.cuda.is_available() else ["cpu"])
    strength = st.slider("Strength", 0.1, 1.0, 0.75, 0.05)
    guidance = st.slider("Guidance Scale", 1.0, 20.0, 7.5, 0.5)
    steps = st.slider("Steps", 10, 50, 30, 5)
    
    nsfw_mode = st.toggle("🔥 NSFW / Explicit Mode", value=True)
    
    style_preset = st.selectbox("Style", [
        "None", "Realistic", "Anime", "Erotic Photography", "NSFW Detailed"
    ])

# Load model (cached after first time)
@st.cache_resource
def load_pipeline():
    pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5",  # Good base, easy to swap for NSFW models
        torch_dtype=torch.float32,
        safety_checker=None  # Disable safety filter for NSFW
    )
    pipe = pipe.to(device)
    return pipe

pipe = load_pipeline()

# Groq for prompt enhancement (optional)
groq_key = os.getenv("GROQ_API_KEY", "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")

col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("📤 Input Image")
    uploaded = st.file_uploader("Upload image", type=["png", "jpg", "jpeg", "webp"])
    if uploaded:
        init_image = Image.open(uploaded).convert("RGB")
        st.image(init_image, caption="Original", use_container_width=True)

    st.subheader("✏️ Prompt")
    prompt = st.text_area("Describe the transformation", 
                         placeholder="seductive naked woman lying on silk sheets, detailed skin, erotic pose...", 
                         height=120)
    
    neg_prompt = st.text_area("Negative Prompt", 
                             "blurry, deformed, ugly, bad anatomy, watermark, text", 
                             height=70)

    enhance = st.button("Enhance Prompt with Groq")
    generate = st.button("🚀 Generate", type="primary")

with col2:
    st.subheader("🖼️ Result")
    result_placeholder = st.empty()

    if enhance and prompt and groq_key:
        with st.spinner("Enhancing..."):
            try:
                client = Groq(api_key=groq_key)
                resp = client.chat.completions.create(
                    model="llama3-70b-8192",
                    messages=[{"role": "user", "content": f"Make this prompt much more detailed and erotic for Stable Diffusion img2img: {prompt}"}],
                    temperature=0.8
                )
                prompt = resp.choices[0].message.content
                st.success("Enhanced!")
                st.write(prompt)
            except:
                st.error("Groq enhancement failed (using original prompt)")

    if generate and uploaded and prompt:
        with result_placeholder.container():
            with st.spinner("Generating on your machine... (can take 10-60s depending on CPU/GPU)"):
                try:
                    final_prompt = prompt
                    if nsfw_mode:
                        final_prompt += ", explicit nudity, detailed anatomy, nsfw, erotic, sensual"
                    if style_preset != "None":
                        final_prompt += f", {style_preset.lower()} style"

                    start = time.time()
                    result = pipe(
                        prompt=final_prompt,
                        image=init_image,
                        strength=strength,
                        guidance_scale=guidance,
                        num_inference_steps=steps,
                        negative_prompt=neg_prompt
                    ).images[0]
                    
                    elapsed = round(time.time() - start, 1)
                    
                    st.image(result, caption=f"Generated in {elapsed}s", use_container_width=True)
                    
                    buf = io.BytesIO()
                    result.save(buf, format="PNG")
                    st.download_button("⬇️ Download", buf.getvalue(), "nsfw_result.png", "image/png")
                    
                    st.success(f"✅ Done in {elapsed} seconds!")
                except Exception as e:
                    st.error(f"Error: {e}")

st.caption("First run downloads the model (~4-7GB). After that it's fully offline and unlimited.")
