
try:
    import gradio as gr
    from pii_audio_anonimization import pii_audio_anonimization
    from modules.Presidio import Presidio
    print("Imports successful")
except ImportError as e:
    print(f"Import failed: {e}")
except Exception as e:
    print(f"An error occurred: {e}")
