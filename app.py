import streamlit as st
from PIL import Image
import requests
from io import BytesIO
import re
import time

st.set_page_config(page_title="Uncensored NSFW Chatbot", page_icon="🖤", layout="centered")

if "messages" not in st.session_state:
    st.session_state.messages = []

def enhance_prompt(raw_prompt: str) -> str:
    prompt = re.sub(r'\s+', ' ', raw_prompt.strip())
    nsfw_boost = "masterpiece, best quality, ultra detailed, realistic skin texture, detailed anatomy, sharp focus, 8k"
    if len(prompt.split()) < 12:
        prompt += f", {nsfw_boost}"
    return prompt.strip()

# ====================== HUGGING FACE PUBLIC ENDPOINT ======================
def generate_image(prompt: str, num_images: int = 1):
    enhanced = enhance_prompt(prompt)
    st.info(f"**Prompt:** {enhanced[:180]}...")

    with st.spinner("🎨 Generating uncensored NSFW image (Hugging Face)..."):
        for attempt in range(4):   # More retries
            try:
                API_URL = "https://api-inference.huggingface.co/models/black-forest-labs/FLUX.1-schnell"
                
                payload = {
                    "inputs": enhanced,
                    "parameters": {
                        "num_inference_steps": 4,
                        "height": 768,
                        "width": 768
                    }
                }
                
                response = requests.post(API_URL, json=payload, timeout=60)
                
                if response.status_code == 200:
                    image = Image.open(BytesIO(response.content))
                    st.image(image, caption=f"✅ NSFW Image {attempt+1}", use_column_width=True)
                    
                    buf = BytesIO()
                    image.save(buf, format="PNG")
                    st.download_button(
                        "⬇️ Download Image",
                        buf.getvalue(),
                        f"nsfw_image_{attempt+1}.png",
                        "image/png",
                        use_container_width=True
                    )
                    st.divider()
                    return True
                
                elif response.status_code == 503:
                    st.warning(f"Model is loading... Waiting {5 + attempt*3} seconds...")
                    time.sleep(5 + attempt*3)
                else:
                    st.warning(f"Attempt {attempt+1} failed ({response.status_code}). Retrying...")
                    time.sleep(7)
                    
            except Exception as e:
                st.warning(f"Attempt {attempt+1} error. Retrying...")
                time.sleep(7)
        
        st.error("❌ Service is busy right now. Try again in 1-2 minutes with a simpler prompt.")
        return False

# ====================== UI ======================
st.title("🖤 Fully Uncensored NSFW Chatbot")
st.caption("Free FLUX • Nudity Allowed • No Limits")

with st.sidebar:
    num_images = st.slider("Number of images", 1, 2, 1)   # Limited for speed
    if st.button("🗑️ Clear Chat"):
        st.session_state.messages = []
        st.rerun()

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if prompt := st.chat_input("Describe anything (be explicit - nudity ok)..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        st.write("Generating...")
        generate_image(prompt, num_images)
    
    st.session_state.messages.append({"role": "assistant", "content": "Image generated."})
