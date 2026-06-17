import streamlit as st
import replicate
import requests
from PIL import Image
import io
import base64
import time

# For Groq LLM integration (prompt enhancement)
from groq import Groq

# ── Page config ──────────────────────────────────────────────
st.set_page_config(
    page_title="NSFW img2img AI Chat Studio",
    page_icon="🎨🔥",
    layout="wide"
)

# ── Custom CSS ───────────────────────────────────────────────
st.markdown("""
<style>
    body { background-color: #0f0f13; }
    .stApp { background-color: #0f0f13; }
    h1 { color: #c084fc; font-family: 'Georgia', serif; }
    .stButton>button {
        background: linear-gradient(135deg, #7c3aed, #c084fc);
        color: white;
        border: none;
        border-radius: 10px;
        padding: 0.6rem 2rem;
        font-size: 1rem;
        font-weight: 600;
        width: 100%;
    }
    .stButton>button:hover {
        background: linear-gradient(135deg, #6d28d9, #a855f7);
        transform: scale(1.02);
    }
    .stSlider>div>div { color: #c084fc; }
    label { color: #d8b4fe !important; }
    .stTextArea textarea, .stTextInput input {
        background-color: #1e1b2e;
        color: #f3e8ff;
        border: 1px solid #7c3aed;
        border-radius: 8px;
    }
    .stSelectbox>div>div {
        background-color: #1e1b2e;
        color: #f3e8ff;
        border: 1px solid #7c3aed;
    }
    .result-box {
        background: #1e1b2e;
        border: 1px solid #7c3aed;
        border-radius: 12px;
        padding: 1rem;
        margin-top: 1rem;
    }
    .tip-box {
        background: #2d1b4e;
        border-left: 4px solid #c084fc;
        padding: 0.75rem 1rem;
        border-radius: 6px;
        color: #e9d5ff;
        font-size: 0.9rem;
        margin-bottom: 1rem;
    }
    .chat-message {
        padding: 1rem;
        border-radius: 12px;
        margin-bottom: 1rem;
    }
    .user-msg { background: #3b2a5e; }
    .ai-msg { background: #1e3a5f; }
</style>
""", unsafe_allow_html=True)

# ── Header ───────────────────────────────────────────────────
st.markdown("# 🎨🔥 NSFW img2img AI Chat Studio")
st.markdown("**Upload image • Chat with AI • Generate uncensored transformations**")
st.markdown("""
💡 **NSFW Allowed** - No content restrictions. Use explicit prompts freely.  
**Tip:** Lower strength = subtle changes. Higher strength = dramatic transformation.
""", unsafe_allow_html=True)

# ── Sidebar — settings ───────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ Settings")
    api_key = st.text_input("Replicate API Key", type="password", 
                           placeholder="r8_xxxxxxxxxxxx", value="")
    groq_key = st.text_input("Groq API Key (for smart prompts)", type="password", 
                            placeholder="gsk_...", value="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")
    st.caption("Get Replicate key at [replicate.com](https://replicate.com)")
    st.divider()
    
    model_choice = st.selectbox("Base Model", [
        "SDXL (General - NSFW friendly)",
        "Realistic Vision (Photorealistic)",
        "Animagine XL (Anime)",
        "DreamShaper XL (Fantasy)",
    ])
    
    nsfw_mode = st.toggle("🔥 Enable Full NSFW Mode", value=True, 
                         help="Allows explicit nudity, sexual content, etc. No filters.")
    
    strength = st.slider("Transformation Strength", 0.1, 1.0, 0.75, 0.05, 
                         help="Higher = more creative / dramatic change")
    guidance = st.slider("Guidance Scale", 1.0, 20.0, 8.5, 0.5)
    steps = st.slider("Inference Steps", 10, 50, 35, 5)
    
    st.divider()
    st.markdown("**Style Preset**")
    style_preset = st.selectbox("Apply style", [
        "None", "Cinematic", "Anime", "Oil Painting", "Watercolor", 
        "Cyberpunk", "Fantasy Art", "Comic Book", "Erotic", "NSFW Explicit"
    ])

# ── Model mapping ────────────────────────────────────────────
MODEL_MAP = {
    "SDXL (General - NSFW friendly)": "stability-ai/sdxl:7762fd07cf82c948538e41f4d1550d0d54f1a40f47ac8b2a37fbbf9e3be2be98",
    "Realistic Vision (Photorealistic)": "lucataco/realistic-vision-v5-img2img:latest",
    "Animagine XL (Anime)": "fofr/animatediff-lightning:latest",
    "DreamShaper XL (Fantasy)": "lucataco/dreamshaper-xl-turbo:latest",
}

STYLE_PROMPTS = {
    "None": "",
    "Cinematic": ", cinematic lighting, film grain, anamorphic lens, dramatic atmosphere",
    "Anime": ", anime style, detailed cel shading, vibrant colors, studio ghibli influence",
    "Oil Painting": ", oil painting, thick brush strokes, masterpiece, classical art",
    "Watercolor": ", watercolor painting, soft edges, translucent layers, artistic",
    "Cyberpunk": ", cyberpunk, neon lights, rain soaked streets, futuristic, blade runner",
    "Fantasy Art": ", epic fantasy art, magical atmosphere, detailed concept art",
    "Comic Book": ", comic book panel, bold ink lines, halftone shading, dynamic angles",
    "Erotic": ", seductive pose, sensual lighting, erotic atmosphere",
    "NSFW Explicit": ", explicit nudity, detailed anatomy, sexual content, aroused expression",
}

# ── Groq Client for prompt enhancement ───────────────────────
def enhance_prompt_with_groq(base_prompt, nsfw_enabled):
    if not groq_key:
        return base_prompt
    try:
        client = Groq(api_key=groq_key)
        system_msg = """You are an expert NSFW image prompt engineer.
        Enhance the user's prompt for img2img. Make it highly detailed, artistic, and explicit if requested.
        Keep it under 300 tokens. Focus on visual descriptors."""
        
        user_msg = f"Original prompt: {base_prompt}\nNSFW mode: {'enabled - be explicit with nudity/sex if it fits' if nsfw_enabled else 'disabled'}"
        
        chat_completion = client.chat.completions.create(
            messages=[
                {"role": "system", "content": system_msg},
                {"role": "user", "content": user_msg}
            ],
            model="llama-3.3-70b-versatile",
            temperature=0.7,
            max_tokens=250
        )
        enhanced = chat_completion.choices[0].message.content.strip()
        return enhanced if enhanced else base_prompt
    except Exception as e:
        st.warning(f"Groq enhancement failed: {e}. Using original prompt.")
        return base_prompt

# ── Main layout ──────────────────────────────────────────────
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

col1, col2 = st.columns([1, 1], gap="large")

with col1:
    st.markdown("### 📤 Input Image")
    uploaded = st.file_uploader("Upload starting image (supports nudity)", 
                               type=["png", "jpg", "jpeg", "webp"])
    if uploaded:
        img = Image.open(uploaded)
        st.image(img, caption="Original Image", use_container_width=True)
        st.caption(f"Size: {img.width}×{img.height}px")

    st.markdown("### 💬 Chat / Prompt")
    prompt = st.text_area("Describe the transformation or chat with AI", 
                         placeholder="A beautiful nude woman with long flowing hair standing under moonlight in a mystical forest, highly detailed, sensual pose...",
                         height=120, key="prompt_input")
    
    if st.button("✨ Enhance Prompt with AI"):
        with st.spinner("Thinking..."):
            enhanced = enhance_prompt_with_groq(prompt, nsfw_mode)
            st.session_state.enhanced_prompt = enhanced
            st.success("Enhanced!")
            st.text_area("Enhanced Prompt", value=enhanced, height=80)
    
    neg_prompt = st.text_area("Negative Prompt", 
                             value="blurry, low quality, deformed, ugly, extra limbs, watermark, text, censored",
                             height=70)
    
    generate_btn = st.button("🚀 Generate Image", type="primary")

with col2:
    st.markdown("### 🖼️ Generation Result + History")
    result_placeholder = st.empty()
    
    # Show chat history
    for msg in st.session_state.chat_history[-5:]:  # last 5
        if msg["role"] == "user":
            st.markdown(f'<div class="chat-message user-msg">🧑 **You:** {msg["content"]}</div>', unsafe_allow_html=True)
        else:
            st.image(msg["image"], caption=msg["caption"], use_container_width=True)
    
    if generate_btn:
        if not api_key:
            result_placeholder.error("Please enter Replicate API key.")
        elif not uploaded:
            result_placeholder.error("Upload an image first.")
        elif not prompt.strip():
            result_placeholder.error("Enter a prompt.")
        else:
            final_prompt = prompt + STYLE_PROMPTS.get(style_preset, "")
            if nsfw_mode and "NSFW Explicit" not in style_preset:
                final_prompt += ", explicit, uncensored, nudity allowed"
            
            # Enhance with Groq if enabled
            if "enhanced_prompt" in st.session_state and st.session_state.get("enhanced_prompt"):
                final_prompt = st.session_state.enhanced_prompt + STYLE_PROMPTS.get(style_preset, "")
            
            with result_placeholder.container():
                with st.spinner("🎨 Generating uncensored image..."):
                    try:
                        client = replicate.Client(api_token=api_key)
                        
                        # Prepare image
                        buf = io.BytesIO()
                        img_copy = Image.open(uploaded).convert("RGB")
                        img_copy.save(buf, format="PNG")
                        b64 = base64.b64encode(buf.getvalue()).decode()
                        data_uri = f"data:image/png;base64,{b64}"
                        
                        start = time.time()
                        
                        model_id = MODEL_MAP.get(model_choice, MODEL_MAP["SDXL (General - NSFW friendly)"])
                        
                        input_params = {
                            "image": data_uri,
                            "prompt": final_prompt,
                            "negative_prompt": neg_prompt,
                            "prompt_strength": strength,
                            "guidance_scale": guidance,
                            "num_inference_steps": steps,
                            "num_outputs": 1,
                        }
                        
                        output = client.run(model_id, input=input_params)
                        
                        elapsed = round(time.time() - start, 1)
                        
                        if output:
                            img_url = output[0] if isinstance(output, list) else output
                            resp = requests.get(str(img_url))
                            result_img = Image.open(io.BytesIO(resp.content))
                            
                            st.image(result_img, caption=f"Generated in {elapsed}s | Model: {model_choice}", 
                                    use_container_width=True)
                            
                            # Save to history
                            st.session_state.chat_history.append({
                                "role": "user",
                                "content": prompt[:100] + "..." if len(prompt)>100 else prompt
                            })
                            st.session_state.chat_history.append({
                                "role": "assistant",
                                "image": result_img,
                                "caption": f"NSFW Transform - {elapsed}s"
                            })
                            
                            buf2 = io.BytesIO()
                            result_img.save(buf2, format="PNG")
                            st.download_button(
                                "⬇️ Download Image",
                                data=buf2.getvalue(),
                                file_name=f"nsfw_img2img_{int(time.time())}.png",
                                mime="image/png"
                            )
                            st.success(f"✅ Done in {elapsed}s! NSFW mode: {'ON' if nsfw_mode else 'OFF'}")
                            
                    except Exception as e:
                        st.error(f"Generation error: {str(e)}")
    else:
        result_placeholder.markdown("""
        <div style="text-align:center; padding:3rem; color:#9ca3af;">
            <h2>🖼️ Your result will appear here</h2>
            <p>Upload image + write detailed NSFW prompt → Generate</p>
        </div>
        """, unsafe_allow_html=True)

# ── Footer ───────────────────────────────────────────────────
st.divider()
st.markdown("**NSFW img2img AI Chat Studio** • Powered by Replicate + Groq • No content limits", unsafe_allow_html=True)
