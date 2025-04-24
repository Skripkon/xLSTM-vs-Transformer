import streamlit as st
from llm_trainer import LLMTrainer
import torch
from transformers import GPT2Config, GPT2LMHeadModel

# Initialize model outside of main to avoid event loop issues
@st.cache_resource
def init_model():
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

def main():
    st.set_page_config(
        page_title="RubbishAI",
        page_icon="📝",
        layout="centered"
    )
    
    st.title("RubbishAI")
    st.write("Enter your text below and the AI will continue it.")

    # Load the model
    trainer = init_model()
    
    if trainer is None:
        st.error("Could not initialize the model. Please check the checkpoint path and try again.")
        return

    # Text input
    user_input = st.text_area("Enter your text:", height=150)
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
                        length=32
                    )
                    for i in range(n_return_sequences):
                        continuations[i] = continuations[i][len(user_input):].strip()

                    # Display the result
                    st.markdown("### Generated Continuations:")
                    # Create 4 columns for the continuations
                    cols = st.columns(n_return_sequences)
                    for idx, continuation in enumerate(continuations):
                        # Trim the text to the last period
                        last_dot_index = continuation.rfind('.')
                        if last_dot_index != -1:
                            trimmed_text = continuation[:last_dot_index + 1]
                        else:
                            trimmed_text = continuation
                            
                        with cols[idx]:
                            st.text_area(f"Continuation {idx + 1}", trimmed_text, height=300, disabled=True)
            except Exception as e:
                st.error(f"Error during generation: {str(e)}")

if __name__ == "__main__":
    main()
