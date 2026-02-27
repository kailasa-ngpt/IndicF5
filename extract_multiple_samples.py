from datasets import load_dataset
import soundfile as sf
import os
import re

print("Extracting 10 Tamil samples for review...")
ds = load_dataset("kailasa-ngpt/tamil-ktkv-1000", split="train", token="YOUR_HF_TOKEN")

os.makedirs("custom_prompts/review", exist_ok=True)
count = 0

def is_pure_tamil(text):
    cleaned = re.sub(r'[ \.\,\!\?\-\'\"]', '', text)
    if not cleaned: return False
    return all('\u0B80' <= c <= '\u0BFF' for c in cleaned)

for idx in range(len(ds)):
    sample = ds[idx]
    if "audio" not in sample or sample["audio"] is None:
        continue
        
    # We want varied samples, not just the longest ones
    ref_text = sample.get("text", sample.get("sentence", sample.get("transcript", "")))
    if not is_pure_tamil(ref_text):
        continue
        
    sr = sample["audio"]["sampling_rate"]
    audio_array = sample["audio"]["array"]
    
    # Needs to be at least 10 seconds long
    duration = len(audio_array) / sr
    if duration < 10.0:
        continue
        
    # Clip to exactly 10 seconds
    num_samples = int(10.0 * sr)
    clipped = audio_array[:num_samples]
    
    # Save it
    filename = f"sample_{idx}.wav"
    sf.write(f"custom_prompts/review/{filename}", clipped, sr)
    with open(f"custom_prompts/review/transcript_{idx}.txt", "w") as f:
        f.write(ref_text)
        
    print(f"Saved {filename}")
    count += 1
    
    if count >= 10:
        break

print("All 10 samples extracted successfully!")
