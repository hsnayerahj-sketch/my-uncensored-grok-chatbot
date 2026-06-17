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

# ====================== BEST FREE OPTION RIGHT NOW ======================
def generate_image(prompt: str, num_images: int = 1):
    enhanced = enhance_prompt(prompt)
    st.info(f"**Prompt:** {enhanced[:180]}...")

    with st.spinner("🎨 Trying to generate uncensored image..."):
        success = False
        for attempt in range(3):  # Retry up to 3 times
            try:
                response = requests.post(
                    "https://api.puter.com/ai/image",
                    json={
                        "model": "black-forest-labs/flux-schnell",
                        "prompt": enhanced,
                        "width": 1024,
                        "height": 1024,
                        "steps": 4,
                        "disable_safety_checker": True
                    },
                    timeout=60
                )

                if response.status_code == 200:
                    data = response.json()
                    image_url = data.get("url") or data.get("image_url")
                    if image_url:
                        img_response = requests.get(image_url, timeout=30)
                        if img_response.status_code == 200:
                            image = Image.open(BytesIO(img_response.content))
                            
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
                            success = True
                            break
                
                else:
                    st.warning(f"Attempt {attempt+1} failed ({response.status_code}). Retrying...")
                    time.sleep(6)
                    
            except Exception as e:
                st.warning(f"Attempt {attempt+1} error. Retrying...")
                time.sleep(6)

        if not success:
            st.error("❌ Service is currently overloaded. Try again in 30-60 seconds or use simpler prompt.")

# ====================== UI ======================
st.title("🖤 Fully Uncensored NSFW Chatbot")
st.caption("Free FLUX • Nudity & No Limits")

with st.sidebar:
    num_images = st.slider("Number of images", 1, 3, 1)
    if st.button("🗑️ Clear Chat"):
        st.session_state.messages = []
        st.rerun()

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if prompt := st.chat_input("Describe anything (be explicit - nudity allowed)..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        st.write("Generating uncensored image...")
        generate_image(prompt, num_images)
    
    st.session_state.messages.append({"role": "assistant", "content": "Image generated."})
