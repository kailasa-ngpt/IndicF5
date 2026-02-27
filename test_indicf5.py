from transformers import AutoModel
import numpy as np
import soundfile as sf
import os

print("Loading AutoModel...")
# Load IndicF5 from Hugging Face
repo_id = "ai4bharat/IndicF5"
model = AutoModel.from_pretrained(repo_id, trust_remote_code=True)

print("Generating speech...")
# Generate speech
audio = model(
    "வணக்கம்! எனக்கு தமிழ் நன்றாக பேச வரும். இது ஒரு சிறப்பான குரல் நகல்.", # "Hello! I can speak Tamil well. This is an excellent voice clone."
    ref_audio_path="custom_prompts/tamil_male_reference_clipped.wav",
    ref_text="சரியாக காளுங்கள். பரம் தற்பம் தரிதா சத்தா வியவஸ்திதம் அஹம் காலே சலச்சித்ரே படலம் சுவயமே வஹி சுவாத்மானம் விஷ்மராமி இஹேக்ஷணஹ ச ஏ வாயுவு காலவ் யவஸ்தையா ஸ்திதஹ காலம் நியஸ்சன் ஜீவன் ஹி"
)

print("Saving output...")
# Normalize and save output
if audio.dtype == np.int16:
    audio = audio.astype(np.float32) / 32768.0

output_path = "samples/namaste.wav"
os.makedirs(os.path.dirname(output_path), exist_ok=True)
sf.write(output_path, np.array(audio, dtype=np.float32), samplerate=24000)
print(f"Audio saved successfully to {output_path}")
