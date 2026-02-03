import gradio as gr
from pii_audio_anonimization import pii_audio_anonimization
import tempfile
import os
import shutil

def anonymize_audio(audio_file, framework, model, anonymizer, token, align_model=None):
    if audio_file is None:
        return None
    
    # Create a temporary output file
    temp_dir = tempfile.mkdtemp()
    output_filename = "anonymized_output.mp3"
    output_path = os.path.join(temp_dir, output_filename)
    
    try:
        pii_audio_anonimization(
            audioFile=audio_file,
            outputFile=output_path,
            transcriptionFramework=framework,
            transcriptionModel=model,
            anonymizerMethod=anonymizer,
            alignModel=align_model,
            token=token
        )
        return output_path
    except Exception as e:
        raise gr.Error(f"An error occurred: {str(e)}")

def update_models(framework):
    if framework == "nemo":
        return gr.Dropdown(choices=["nvidia/canary-1b-v2", "nvidia/parakeet-tdt-0.6b-v3"], value="nvidia/canary-1b-v2", allow_custom_value=False)
    elif framework == "qwen3asr":
        return gr.Dropdown(choices=["Qwen/Qwen3-ASR-1.7B","Qwen/Qwen3-ASR-0.6B"], value="Qwen/Qwen3-ASR-1.7B", allow_custom_value=True)
    else:
        return gr.Dropdown(choices=["large-v3", "large-v2", "medium", "small", "base", "tiny"], value="large-v3", allow_custom_value=True)
        

# Define Gradio interface
with gr.Blocks(title="PII Audio Anonymization") as demo:
    gr.Markdown("# PII Audio Anonymization Tool")
    gr.Markdown("Upload an audio file to detect and anonymize Personally Identifiable Information (PII).")
    
    with gr.Row():
        with gr.Column():
            audio_input = gr.Audio(type="filepath", label="Input Audio")
            
            with gr.Accordion("Advanced Settings", open=False):
                framework_input = gr.Dropdown(
                    choices=["whisperx", "nemo","qwen3asr"], 
                    value="whisperx", 
                    label="Transcription Framework"
                )
                model_input = gr.Dropdown(choices=["large-v3", "large-v2", "medium", "small", "base", "tiny"], value="large-v3", label="Transcription Model", allow_custom_value=True)
                anonymizer_input = gr.Dropdown(choices=["eu-pii-safeguard", "presidio"], value="eu-pii-safeguard", label="Anonymizer Method")
                align_model_input = gr.Textbox(value=None, label="Alignment Model (Optional)")
                token_input = gr.Textbox(label="Hugging Face Token", type="password", placeholder="Required for some models")
            
            anonymize_btn = gr.Button("Anonymize Audio", variant="primary")
            
        with gr.Column():
            audio_output = gr.Audio(label="Anonymized Audio")
    
    anonymize_btn.click(
        fn=anonymize_audio,
        inputs=[audio_input, framework_input, model_input, anonymizer_input, token_input, align_model_input],
        outputs=audio_output
    )
    
    framework_input.change(
        fn=update_models,
        inputs=framework_input,
        outputs=model_input
    )

if __name__ == "__main__":
    demo.launch()
