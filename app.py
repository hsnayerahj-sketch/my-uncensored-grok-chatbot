import streamlit as st
from PIL import Image
import requests
from io import BytesIO
import re
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
        groq_api_key=st.secrets.get("GROQ_API_KEY", ""),  # Use secrets, not hardcoded key
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


def generate_image_hf(prompt: str, num_images: int = 1):
    """
    Uses Hugging Face Serverless Inference API — truly free, no credit card.
    Returns raw image bytes directly — single request, no second fetch needed.
    Sign up: https://huggingface.co/join → Settings → Access Tokens → New Token (read)
    """
    simple_prompt = simplify_prompt(prompt)
    hf_token = st.secrets.get("HF_TOKEN", "")

    if not hf_token:
        st.error("❌ Add HF_TOKEN to your .streamlit/secrets.toml")
        st.code("""
# .streamlit/secrets.toml
HF_TOKEN = "hf_your_token_here"   # huggingface.co → Settings → Access Tokens
GROQ_API_KEY = "GROQ_API_KEY = "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"
""")
        return

    API_URL = "https://api-inference.huggingface.co/models/black-forest-labs/FLUX.1-schnell"
    headers = {"Authorization": f"Bearer {hf_token}"}

    for i in range(num_images):
        with st.spinner(f"🎨 Generating image {i+1}/{num_images}..."):
            try:
                response = requests.post(
                    API_URL,
                    headers=headers,
                    json={"inputs": simple_prompt},
                    timeout=60,
                )

                if response.status_code == 200:
                    # HF returns raw image bytes — load directly, no base64 decode needed
                    image = Image.open(BytesIO(response.content))
                    st.image(image, caption=f"✅ Image {i+1}", use_column_width=True)

                    buf = BytesIO()
                    image.save(buf, format="PNG")
                    st.download_button(
                        label="⬇️ Download",
                        data=buf.getvalue(),
                        file_name=f"flux_{i+1}.png",
                        mime="image/png",
                        use_container_width=True,
                        key=f"dl_{i}_{prompt[:20]}",
                    )
                    if num_images > 1:
                        st.divider()

                elif response.status_code == 503:
                    # Model is loading (cold start) — HF will tell you estimated wait
                    wait = response.json().get("estimated_time", 20)
                    st.warning(f"Model warming up, retrying in {int(wait)}s...")
                    import time
                    time.sleep(min(wait, 30))
                    # Retry once after warm-up
                    retry = requests.post(API_URL, headers=headers,
                                          json={"inputs": simple_prompt}, timeout=60)
                    if retry.status_code == 200:
                        image = Image.open(BytesIO(retry.content))
                        st.image(image, caption=f"✅ Image {i+1}", use_column_width=True)
                    else:
                        st.error(f"Retry failed ({retry.status_code}). Try again in a moment.")
                else:
                    st.error(f"Image {i+1} failed ({response.status_code}): {response.text[:200]}")

            except requests.exceptions.Timeout:
                st.error(f"Image {i+1}: Timed out. HF may be under load — try again.")
            except Exception as e:
                st.error(f"Image {i+1} error: {str(e)[:150]}")


# ====================== UI ======================
st.title("🖤 Uncensored Chatbot + Fast Image Gen")
st.caption("Powered by Groq (LLaMA 3.3) + Hugging Face FLUX.schnell (free)")

with st.sidebar:
    st.header("⚙️ Settings")
    num_images = st.slider("Number of images", 1, 3, 1)
    st.caption("Fewer = faster. Start with 1.")
    if st.button("🗑️ Clear Chat"):
        st.session_state.messages = []
        st.rerun()
    st.divider()
    st.markdown("**Speed tips:**\n- Use 1 image at a time\n- Keep prompts concise\n- Free tier: ~5–10s per image")

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

            # Stream LLM response
            for chunk in model.stream(messages):
                if chunk.content:
                    full_response += chunk.content
                    message_placeholder.markdown(full_response + "▌")

            message_placeholder.markdown(full_response)

        except Exception as e:
            st.error(f"LLM error: {str(e)}")

    st.session_state.messages.append({"role": "assistant", "content": full_response})

    # Trigger image generation after response is saved
    if "GENERATE_IMAGE:" in full_response.upper():
        match = re.search(r"GENERATE_IMAGE:\s*(.+)", full_response, re.IGNORECASE)
        if match:
            img_prompt = match.group(1).strip()
            generate_image_hf(img_prompt, num_images)
