import streamlit as st
from transformers import MarianMTModel, MarianTokenizer
import torch

st.set_page_config(page_title="AI Translator")
st.title("Real-Time AI Translator")

@st.cache_resource
def load_translator():
    model_name = "Helsinki-NLP/opus-mt-en-hi"
    tokenizer = MarianTokenizer.from_pretrained(model_name)
    model = MarianMTModel.from_pretrained(model_name)
    return tokenizer, model

if st.sidebar.button("Force Clear Cache"):
    st.cache_resource.clear()
    st.rerun()

try:
    tokenizer, model = load_translator()

    text_input = st.text_area("Enter English text:", placeholder="Type here...", height=150)

    if st.button("Translate"):
        if text_input.strip():
            with st.spinner("Transformer is processing..."):
                inputs = tokenizer(text_input, return_tensors="pt", padding=True)
                
                with torch.no_grad():
                    translated_tokens = model.generate(**inputs)
                
                result = tokenizer.decode(translated_tokens[0], skip_special_tokens=True)
                
                st.subheader("Hindi Translation:")
                st.success(result)
        else:
            st.warning("Please enter text to translate.")

except Exception as e:
    st.error(f"Error: {e}")
    st.info("Click 'Force Clear Cache' in the sidebar and try again.")