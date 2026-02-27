from datasets import load_dataset
import soundfile as sf
import os
import csv
from tqdm import tqdm

print("Downloading dataset to local fine-tuning format...")
ds = load_dataset("kailasa-ngpt/tamil-ktkv-1000", split="train", token="YOUR_HF_TOKEN")

out_dir = "custom_dataset"
wavs_dir = os.path.join(out_dir, "wavs")
os.makedirs(wavs_dir, exist_ok=True)
metadata_file = os.path.join(out_dir, "metadata.csv")

# We will grab just the first 100 purely Tamil samples to keep the duration and memory low for this test,
# but we can increase it if needed later.
max_samples = 150
count = 0

def is_pure_tamil(text):
    import re
    cleaned = re.sub(r'[ \.\,\!\?\-\'\"]', '', text)
    if not cleaned: return False
    return all('\u0B80' <= c <= '\u0BFF' for c in cleaned)

with open(metadata_file, 'w', encoding='utf-8') as f:
    writer = csv.writer(f, delimiter='|')
    writer.writerow(["audio_file", "text"])
    
    for idx, sample in enumerate(tqdm(ds)):
        if "audio" not in sample or sample["audio"] is None:
            continue
            
        ref_text = sample.get("text", sample.get("sentence", sample.get("transcript", "")))
        if not is_pure_tamil(ref_text):
            continue
            
        audio_array = sample["audio"]["array"]
        sr = sample["audio"]["sampling_rate"]
        
        # Save audio
        wav_name = f"sample_{idx:05d}.wav"
        wav_path = os.path.abspath(os.path.join(wavs_dir, wav_name))
        sf.write(wav_path, audio_array, sr)
        
        # The script `prepare_csv_wavs.py` expects either the raw filename if it assumes `wavs/` dir,
        # or the exact path as written. We will write the relative path to be safe.
        writer.writerow([wav_path, ref_text])
        
        count += 1
        if count >= max_samples:
            break

print(f"\nSaved {count} samples to '{out_dir}/' ready for the formatting script!")
