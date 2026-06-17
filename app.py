import streamlit as st
from PIL import Image
import requests
from io import BytesIO
import re

st.set_page_config(page_title="Uncensored NSFW Chatbot", page_icon="🖤", layout="centered")

if "messages" not in st.session_state:
    st.session_state.messages = []

# Strong NSFW enhancer
def enhance_prompt(raw_prompt: str) -> str:
    prompt = re.sub(r'\s+', ' ', raw_prompt.strip())
    nsfw_boost = "masterpiece, best quality, ultra detailed, realistic skin, detailed anatomy, 8k"
    if len(prompt.split()) < 12:
        prompt += f", {nsfw_boost}"
    return prompt.strip()

# Fast Free NSFW Image Generation
def generate_image(prompt: str, num_images: int = 1):
    enhanced = enhance_prompt(prompt)
    st.info(f"**Generating:** {enhanced[:150]}...")

    with st.spinner("🎨 Generating uncensored NSFW image..."):
        for i in range(num_images):
            try:
                response = requests.post(
                    "https://api.puter.com/ai/image",
                    json={
                        "model": "black-forest-labs/flux-schnell",
                        "prompt": enhanced,
                        "width": 1024,
                        "height": 1024,
                        "steps": 5,
                        "disable_safety_checker": True
                    },
                    timeout=60
                )

                if response.status_code == 200:
                    data = response.json()
                    image_url = data.get("url") or data.get("image_url")
                    if image_url:
                        img = Image.open(BytesIO(requests.get(image_url).content))
                        st.image(img, caption=f"✅ NSFW Image {i+1}", use_column_width=True)
                        
                        buf = BytesIO()
                        img.save(buf, format="PNG")
                        st.download_button(
                            "⬇️ Download Image",
                            buf.getvalue(),
                            f"nsfw_{i+1}.png",
                            "image/png",
                            use_container_width=True
                        )
                        st.divider()
                else:
                    st.error("Service busy. Try again in 10 seconds.")
            except Exception as e:
                st.error(f"Error: {str(e)[:100]}")

# ====================== UI ======================
st.title("🖤 Fully Uncensored NSFW Chatbot")
st.caption("Nudity & NSFW Allowed • No Limits")

with st.sidebar:
    num_images = st.slider("Number of images", 1, 3, 1)
    if st.button("🗑️ Clear Chat"):
        st.session_state.messages = []
        st.rerun()

# Chat
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if prompt := st.chat_input("Describe anything (nudity, nsfw, erotic, etc.)"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        full_response = "Generating your image..."
        st.write(full_response)
        
        generate_image(prompt, num_images)
    
    st.session_state.messages.append({"role": "assistant", "content": full_response})
