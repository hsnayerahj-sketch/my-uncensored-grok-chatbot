import streamlit as st
from openai import OpenAI
import os
from datetime import datetime

st.set_page_config(
    page_title="🔥 Grok Image Generator",
    page_icon="🖼️",
    layout="wide"
)

st.title("🔥 My Grok Image Generator")
st.caption("Text-to-Image • Powered by xAI Grok")

# ===================== SIDEBAR =====================
with st.sidebar:
    st.header("⚙️ Settings")

    api_key = st.text_input(
        "xAI API Key",
        type="password",
        value=os.getenv("XAI_API_KEY", "")
    )

    if st.button("💾 Save API Key"):
        os.environ["XAI_API_KEY"] = api_key
        st.success("Key saved!")

    uncensored_mode = st.toggle("Maximum Mode", value=True)
    num_images = st.slider("Number of Images", 1, 4, 1)

# ===================== CLIENT =====================
if not api_key:
    st.warning("Please enter your xAI API key")
    st.stop()

client = OpenAI(
    base_url="https://api.x.ai/v1",
    api_key=api_key
)

# ===================== SYSTEM PROMPT =====================
system_prompt = "You are an AI image generation assistant."

if uncensored_mode:
    system_prompt = "MAXIMUM MODE: " + system_prompt

# ===================== SESSION =====================
if "messages" not in st.session_state:
    st.session_state.messages = []

# ===================== HISTORY =====================
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

        if "images" in msg:
            for url in msg["images"]:
                st.image(url, use_container_width=True)

# ===================== INPUT =====================
prompt = st.chat_input("Describe the image you want...")

if prompt:
    # Save user message
    st.session_state.messages.append({
        "role": "user",
        "content": prompt
    })

    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        try:
            with st.spinner("Generating images..."):

                response = client.images.generate(
                    model="grok-imagine-image-quality",
                    prompt=prompt,
                    n=num_images
                )

                image_urls = [img.url for img in response.data]

                for url in image_urls:
                    st.image(url, use_container_width=True)

                    st.download_button(
                        "⬇️ Download",
                        data=url,
                        file_name=f"grok_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
                    )

                st.session_state.messages.append({
                    "role": "assistant",
                    "content": "Generated images:",
                    "images": image_urls
                })

        except Exception as e:
            st.error(f"Error: {str(e)}")

            st.session_state.messages.append({
                "role": "assistant",
                "content": f"Error: {str(e)}"
            })
