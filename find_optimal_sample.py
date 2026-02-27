from datasets import load_dataset
import soundfile as sf
import os
import re

print("Analyzing sample lengths for pure Tamil in kailasa-ngpt/tamil-ktkv-1000...")
ds = load_dataset("kailasa-ngpt/tamil-ktkv-1000", split="train", token="YOUR_HF_TOKEN")

# Tamil unicode block is roughly \u0B80-\u0BFF
# Let's write a simple heuristic to ensure it's mostly Tamil letters
def is_pure_tamil(text):
    # Remove spaces and punctuation
    cleaned = re.sub(r'[ \.\,\!\?\-\'\"]', '', text)
    if not cleaned: 
        return False
    # Check if all remaining characters are in the Tamil unicode block
    return all('\u0B80' <= c <= '\u0BFF' for c in cleaned)

ideal_duration = 0
best_idx = -1

for idx in range(len(ds)):
    sample = ds[idx]
    if "audio" not in sample or sample["audio"] is None:
        continue
    
    # User said the previous one had Sanskrit, which was index 152
    if idx == 152: 
        continue
        
    ref_text = sample.get("text", sample.get("sentence", sample.get("transcript", "")))
    if not is_pure_tamil(ref_text):
        continue
    
    sr = sample["audio"]["sampling_rate"]
    num_samples = len(sample["audio"]["array"])
    duration = num_samples / sr
    
    # Let's try to find the longest one that is strictly Tamil
    if duration > ideal_duration:
        ideal_duration = duration
        best_idx = idx

if best_idx != -1:
    print(f"Longest pure Tamil sample duration: {ideal_duration:.2f} seconds at index {best_idx}")
    sample = ds[best_idx]
    os.makedirs("custom_prompts", exist_ok=True)
    out_path = "custom_prompts/tamil_male_reference_optimal.wav"
    sf.write(out_path, sample["audio"]["array"], sample["audio"]["sampling_rate"])

    ref_text = sample.get("text", sample.get("sentence", sample.get("transcript", "")))
    with open("custom_prompts/optimal_transcript.txt", "w") as f:
        f.write(ref_text)
        
    print(f"Saved longest Tamil sample to {out_path}")
    print(f"Transcript: {ref_text}")
    print(f"Reference length: {ideal_duration:.2f} seconds")
else:
    print("Could not find any pure Tamil samples.")
