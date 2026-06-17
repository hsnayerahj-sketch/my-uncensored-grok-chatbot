import streamlit as st
from PIL import Image
import torch
from io import BytesIO
import re

st.set_page_config(page_title="Uncensored NSFW FLUX Chatbot", page_icon="🖤", layout="centered")

@st.cache_resource(show_spinner="Loading Local FLUX.1-Schnell (Uncensored)...")
def load_flux():
    from diffusers import FluxPipeline
    pipe = FluxPipeline.from_pretrained(
        "black-forest-labs/FLUX.1-schnell",
        torch_dtype=torch.bfloat16,
    )
    pipe.enable_model_cpu_offload()
    return pipe

pipe = load_flux()

if "messages" not in st.session_state:
    st.session_state.messages = []

# Strong NSFW prompt enhancer
def enhance_prompt(raw_prompt: str) -> str:
    if not raw_prompt:
        return "beautiful woman, nude, detailed skin, masterpiece"
    
    prompt = re.sub(r'\s+', ' ', raw_prompt.strip())
    
    # Auto add enhancers for NSFW
    nsfw_enhancers = ("masterpiece, best quality, ultra detailed, 8k, realistic skin texture, "
                     "intricate details, sharp focus, cinematic lighting")
    
    if len(prompt.split()) < 15:
        prompt += f", {nsfw_enhancers}"
    
    return prompt.strip()

# ====================== UNCENSORED LOCAL FLUX ======================
def generate_image(prompt: str, num_images: int = 1):
    enhanced_prompt = enhance_prompt(prompt)
    st.info(f"**Prompt:** {enhanced_prompt[:200]}...")

    with st.spinner("🎨 Generating uncensored image (NSFW + Nudity allowed)..."):
        for i in range(num_images):
            try:
                image = pipe(
                    prompt=enhanced_prompt,
                    num_inference_steps=6,
                    guidance_scale=3.5,
                    height=1024,
                    width=1024,
                    max_sequence_length=512,
                ).images[0]

                st.image(image, caption=f"✅ NSFW Image {i+1}", use_column_width=True)

                buf = BytesIO()
                image.save(buf, format="PNG")
                st.download_button(
                    label="⬇️ Download Image",
                    data=buf.getvalue(),
                    file_name=f"nsfw_flux_{i+1}.png",
                    mime="image/png",
                    use_container_width=True
                )
                st.divider()
            except Exception as e:
                st.error(f"Error: {str(e)}")

# ====================== UI ======================
st.title("🖤 Fully Uncensored NSFW FLUX Chatbot")
st.caption("Nudity • NSFW • No Restrictions • Local")

with st.sidebar:
    num_images = st.slider("Number of images", 1, 4, 1)
    if st.button("🗑️ Clear Chat"):
        st.session_state.messages = []
        st.rerun()

uploaded_file = st.file_uploader("Reference image (optional)", type=["png", "jpg", "jpeg", "webp"])
reference_image = Image.open(uploaded_file) if uploaded_file else None
if reference_image:
    st.image(reference_image, caption="Reference Image")

# Chat History
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("images"):
            for img in msg.get("images", []):
                st.image(img)

if prompt := st.chat_input("Describe anything... (nudity, nsfw, no limits)"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    with st.chat_message("user"):
        st.markdown(prompt)
        if reference_image:
            st.image(reference_image)

    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = "Generating uncensored image for you..."
        message_placeholder.markdown(full_response)
        
        try:
            generate_image(prompt, num_images)
        except Exception as e:
            st.error(f"Error: {str(e)}")
    
    st.session_state.messages.append({"role": "assistant", "content": full_response})
