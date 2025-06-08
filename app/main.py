from typing import Literal

import streamlit as st
from omegaconf import OmegaConf
from dacite import Config as DaciteConfig, from_dict
from transformers import GPT2Config, GPT2LMHeadModel

from llm_trainer import LLMTrainer
from xlstm import xLSTMLMModel, xLSTMLMModelConfig


# Initialize model outside of main to avoid event loop issues
@st.cache_resource
def init_model(model_name: Literal["xLSTM", "GPT2"]):
    match model_name:
        case "GPT2":
            config = GPT2Config(
                vocab_size=50304,
                n_positions=256,
                n_embd=768,
                n_layer=12,
                n_head=12,
                activation_function="gelu"
            )
            gpt2 = GPT2LMHeadModel(config)
            trainer = LLMTrainer(model=gpt2, model_returns_logits=False)
            try:
                trainer.load_checkpoint("research/checkpoints_gpt/cp_3999.pth")
                return trainer
            except Exception as e:
                st.error(f"Failed to load model checkpoint: {str(e)}")
                return None
        case "xLSTM":
            cfg = OmegaConf.load("research/xlstm_config.yaml")
            cfg = from_dict(data_class=xLSTMLMModelConfig, data=OmegaConf.to_container(cfg), config=DaciteConfig(strict=True))
            xLSTM = xLSTMLMModel(cfg)
            trainer = LLMTrainer(model=xLSTM, model_returns_logits=True)
            try:
                trainer.load_checkpoint("research/checkpoints_xlstm/cp_9999.pth")
                return trainer
            except Exception as e:
                st.error(f"Failed to load model checkpoint: {str(e)}")
                return None
        case _:
            raise ValueError(f"{model_name} is invalid model! Valid models are `xLSTM` and `GPT2`")

def main(model_name: Literal["xLSTM", "GPT2"]):
    st.set_page_config(
        page_title="HSEAI",
        page_icon="📝",
        layout="centered"
    )
    
    st.title("HSEAI")
    st.write("Enter your text below and the AI will continue it.")

    # Load the model
    trainer = init_model(model_name=model_name)
    
    if trainer is None:
        st.error("Could not initialize the model. Please check the checkpoint path and try again.")
        return

    # Text input
    user_input = st.text_area("Enter your text:", height=70)
    n_return_sequences = 3

    # Generate button
    if st.button("Generate Continuation"):
        if not user_input:
            st.warning("Please enter some text first.")
        else:
            try:
                with st.spinner("Generating..."):
                    # Generate continuation
                    continuations = trainer.generate_text(
                        prompt=user_input,
                        n_return_sequences=n_return_sequences,
                        length=64
                    )
                    for i in range(n_return_sequences):
                        continuations[i] = continuations[i][len(user_input):].strip()

                    # Create 4 columns for the continuations
                    cols = st.columns(n_return_sequences)
                    for idx, continuation in enumerate(continuations):
                        with cols[idx]:
                            st.markdown(f"""
                            <div style="background-color:#0e1117; padding:10px; border-radius:8px; height:300px; overflow:auto;">
                                <p style="color:white; white-space:pre-wrap;">{continuation + "..."}</p>
                            </div>
                            """, unsafe_allow_html=True)

            except Exception as e:
                st.error(f"Error during generation: {str(e)}")

if __name__ == "__main__":  # run `streamlit run app/main.py` FROM ROOT
    main(model_name="xLSTM")
