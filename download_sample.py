from datasets import load_dataset
import soundfile as sf
import os

print("Loading dataset kailasa-ngpt/tamil-ktkv-1000...")
ds = load_dataset("kailasa-ngpt/tamil-ktkv-1000", split="train", token="YOUR_HF_TOKEN")
print(f"Dataset loaded. Number of rows: {len(ds)}")

sample = ds[0]
print("Keys available in the dataset:", sample.keys())

# Extract text
ref_text = ""
if "text" in sample:
    ref_text = sample["text"]
elif "sentence" in sample:
    ref_text = sample["sentence"]
elif "transcript" in sample:
    ref_text = sample["transcript"]

print("Reference Text extracted:", ref_text)

# Save the audio file
os.makedirs("custom_prompts", exist_ok=True)
audio_data = sample['audio']['array']
sr = sample['audio']['sampling_rate']
ref_audio_path = "custom_prompts/tamil_male_reference.wav"

sf.write(ref_audio_path, audio_data, sr)
print(f"Reference audio saved to {ref_audio_path}")

print("---")
print(f"REF_AUDIO_PATH='{ref_audio_path}'")
print(f"REF_TEXT='{ref_text}'")
