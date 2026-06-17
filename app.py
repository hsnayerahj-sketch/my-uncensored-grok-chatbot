import streamlit as st
from diffusers import StableDiffusionImg2ImgPipeline
import torch
from PIL import Image
import io
import time
from groq import Groq
import os

# Page Config
st.set_page_config(
    page_title="Free NSFW img2img Studio",
    page_icon="🎨",
    layout="wide"
)

st.title("🎨 Free Local NSFW img2img Studio")
st.markdown("**100% Free • Unlimited • No Payment • Fully Uncensored**")
st.caption("First time will take 3-8 minutes to download the model")

# Sidebar
with st.sidebar:
    st.header("⚙️ Settings")
    
    use_gpu = torch.cuda.is_available()
    device = st.selectbox("Device", ["cuda"] if use_gpu else ["cpu"], 
                         help="cuda is much faster if available")
    
    strength = st.slider("Transformation Strength", 0.1, 1.0, 0.75, 0.05, 
                        help="Higher = more change (good for nudity)")
    guidance = st.slider("Guidance Scale", 1.0, 20.0, 7.5, 0.5)
    steps = st.slider("Inference Steps", 10, 50, 30, 5)
    
    nsfw_mode = st.toggle("🔥 Enable NSFW / Explicit Mode", value=True)
    
    style_preset = st.selectbox("Style Preset", [
        "None", "Realistic", "Erotic Photography", "Anime", 
        "Sensual Lighting", "NSFW Detailed"
    ])

# Model Loading (cached)
@st.cache_resource(show_spinner="Downloading AI Model... (First time only - ~4-7GB)")
def load_pipeline():
    pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5",
        torch_dtype=torch.float32,
        safety_checker=None,      # Important: Disable for NSFW
        use_safetensors=True
    )
    pipe = pipe.to(device)
    return pipe

pipe = load_pipeline()

# Groq Key
groq_key = os.getenv("GROQ_API_KEY", "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")

col1, col2 = st.columns([1, 1], gap="large")

with col1:
    st.subheader("📤 Input Image")
    uploaded = st.file_uploader("Upload your starting image", 
                               type=["png", "jpg", "jpeg", "webp"])
    
    if uploaded:
        init_image = Image.open(uploaded).convert("RGB")
        st.image(init_image, caption="Original Image", use_container_width=True)

    st.subheader("✏️ Prompt")
    prompt = st.text_area(
        "Describe what you want", 
        placeholder="A beautiful naked woman with long hair, seductive pose on bed, detailed skin, erotic atmosphere...",
        height=130
    )
    
    neg_prompt = st.text_area("Negative Prompt", 
                             "blurry, low quality, deformed, ugly, bad anatomy, extra limbs, watermark, text",
                             height=80)

    col_a, col_b = st.columns(2)
    with col_a:
        enhance_btn = st.button("✨ Enhance Prompt (Groq)")
    with col_b:
        generate_btn = st.button("🚀 Generate Image", type="primary")

with col2:
    st.subheader("🖼️ Result")
    result_placeholder = st.empty()

    if enhance_btn and prompt.strip():
        if groq_key:
            with st.spinner("Enhancing prompt with AI..."):
                try:
                    client = Groq(api_key=groq_key)
                    response = client.chat.completions.create(
                        model="llama3-70b-8192",
                        messages=[{
                            "role": "system",
                            "content": "You are an expert erotic NSFW prompt engineer for Stable Diffusion."
                        }, {
                            "role": "user",
                            "content": f"Make this prompt much more detailed, vivid and erotic: {prompt}"
                        }],
                        temperature=0.85,
                        max_tokens=350
                    )
                    prompt = response.choices[0].message.content.strip()
                    st.success("✅ Prompt Enhanced!")
                    st.write("**Enhanced Prompt:**", prompt)
                except Exception as e:
                    st.error(f"Groq failed: {e}")
        else:
            st.error("Groq key not found")

    if generate_btn:
        if not uploaded:
            st.error("Please upload an image first")
        elif not prompt.strip():
            st.error("Please enter a prompt")
        else:
            with result_placeholder.container():
                with st.spinner("Generating image... (this can take 15-60 seconds)"):
                    try:
                        final_prompt = prompt
                        if nsfw_mode:
                            final_prompt += ", explicit nudity, detailed pussy, detailed breasts, erotic, sensual, nsfw, masterpiece"
                        if style_preset != "None":
                            final_prompt += f", {style_preset.lower()}"

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
                        
                        st.image(result, caption=f"✅ Generated in {elapsed} seconds", use_container_width=True)
                        
                        buf = io.BytesIO()
                        result.save(buf, format="PNG")
                        st.download_button(
                            "⬇️ Download Image",
                            buf.getvalue(),
                            file_name=f"nsfw_result_{int(time.time())}.png",
                            mime="image/png"
                        )
                        
                    except Exception as e:
                        st.error(f"Generation Error: {str(e)}")

st.divider()
st.caption("Running on GitHub Codespaces • Model: Stable Diffusion v1.5 • Fully Free & Uncensored")
