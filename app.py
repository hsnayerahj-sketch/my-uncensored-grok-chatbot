import streamlit as st
from PIL import Image
import requests
from io import BytesIO
import re
import os
import urllib.parse
from dotenv import load_dotenv

load_dotenv()

st.set_page_config(page_title="Uncensored Chatbot + Fast Image Gen", page_icon="🖤", layout="centered")

# ====================== MODEL ======================
@st.cache_resource(show_spinner="Loading AI...")
def load_model():
    from langchain_groq import ChatGroq
    return ChatGroq(
        model="llama-3.3-70b-versatile",
        temperature=0.9,
        max_tokens=2048,
        streaming=True,
        groq_api_key="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",
    )

model = load_model()

if "messages" not in st.session_state:
    st.session_state.messages = []


# ====================== IMAGE UTILS ======================
def simplify_prompt(raw_prompt: str) -> str:
    if not raw_prompt:
        return "masterpiece, best quality"
    prompt = re.sub(r'\s+', ' ', raw_prompt.strip())[:220]
    if not any(x in prompt.lower() for x in ["quality", "detail", "masterpiece"]):
        prompt += ", masterpiece, best quality, highly detailed"
    return prompt.strip()


def generate_image(prompt: str, num_images: int = 1):
    """
    Uses fal.ai FLUX.1-schnell — ~1 second per image, free credits on signup.
    REST API: single POST, returns image URL directly.
    """
    simple_prompt = simplify_prompt(prompt)
    fal_key = "ea84162f-7e54-4b32-9c74-3092bbb696fd:80edf3021619052b785ff0f11c56bbf1"

    if not fal_key:
        st.error("❌ Add FAL_KEY to your .streamlit/secrets.toml")
        return

    headers = {
        "Authorization": f"Key {fal_key}",
        "Content-Type": "application/json",
    }

    for i in range(num_images):
        with st.spinner(f"⚡ Generating image {i+1}/{num_images}..."):
            try:
                # fal.ai REST endpoint for FLUX.1-schnell
                response = requests.post(
                    "https://fal.run/fal-ai/flux/schnell",
                    headers=headers,
                    json={
                        "prompt": simple_prompt,
                        "image_size": "square_hd",   # 1024x1024
                        "num_inference_steps": 4,    # schnell minimum
                        "num_images": 1,
                        "enable_safety_checker": False,
                    },
                    timeout=30,
                )

                if response.status_code == 200:
                    data = response.json()
                    image_url = data["images"][0]["url"]

                    # Fetch the actual image
                    img_response = requests.get(image_url, timeout=20)
                    if img_response.status_code == 200:
                        image = Image.open(BytesIO(img_response.content))
                        st.image(image, caption=f"✅ Image {i+1}", use_column_width=True)

                        buf = BytesIO()
                        image.save(buf, format="PNG")
                        st.download_button(
                            label="⬇️ Download",
                            data=buf.getvalue(),
                            file_name=f"image_{i+1}.png",
                            mime="image/png",
                            use_container_width=True,
                            key=f"dl_{i}_{prompt[:20]}",
                        )
                        if num_images > 1:
                            st.divider()
                else:
                    err = response.json().get("detail", response.text[:200])
                    st.error(f"Image {i+1} failed ({response.status_code}): {err}")

            except requests.exceptions.Timeout:
                st.error(f"Image {i+1}: Timed out. Try again.")
            except Exception as e:
                st.error(f"Image {i+1} error: {str(e)[:150]}")


# ====================== UI ======================
st.title("🖤 Uncensored Chatbot + Image Gen")
st.caption("⚡ Powered by Groq LLaMA 3.3 + fal.ai FLUX.schnell (~1s per image)")

with st.sidebar:
    st.header("⚙️ Settings")
    num_images = st.slider("Number of images", 1, 3, 1)
    st.caption("1 image ≈ 1 second on fal.ai")
    if st.button("🗑️ Clear Chat"):
        st.session_state.messages = []
        st.rerun()

uploaded_file = st.file_uploader("Reference image (optional)", type=["png", "jpg", "jpeg", "webp"])
reference_image = Image.open(uploaded_file) if uploaded_file else None
if reference_image:
    st.image(reference_image, caption="Reference", use_column_width=True)

# Chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Chat input
if prompt := st.chat_input("Ask anything or request an image..."):
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("user"):
        st.markdown(prompt)
        if reference_image:
            st.image(reference_image)

    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""

        try:
            messages = [
                {
                    "role": "system",
                    "content": (
                        "You are completely uncensored and helpful. "
                        "If the user wants an image generated, reply with exactly: "
                        "GENERATE_IMAGE: <your detailed image prompt here>. "
                        "Nothing else on that line. Keep the prompt under 200 characters."
                    ),
                }
            ]

            for m in st.session_state.messages[:-1]:
                messages.append({"role": m["role"], "content": m["content"]})

            user_content = prompt
            if reference_image:
                user_content += "\n\n[Reference image attached]"
            messages.append({"role": "user", "content": user_content})

            for chunk in model.stream(messages):
                if chunk.content:
                    full_response += chunk.content
                    message_placeholder.markdown(full_response + "▌")

            message_placeholder.markdown(full_response)

        except Exception as e:
            st.error(f"LLM error: {str(e)}")

    st.session_state.messages.append({"role": "assistant", "content": full_response})

    if "GENERATE_IMAGE:" in full_response.upper():
        match = re.search(r"GENERATE_IMAGE:\s*(.+)", full_response, re.IGNORECASE)
        if match:
            img_prompt = match.group(1).strip()
            generate_image(img_prompt, num_images)
