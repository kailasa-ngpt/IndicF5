import soundfile as sf
import os

in_path = "custom_prompts/tamil_male_reference_optimal.wav"
out_path = "custom_prompts/tamil_male_reference_clipped.wav"

audio, sr = sf.read(in_path)

# 45 seconds is too long for a reference prompt. It can cause out-of-memory errors
# or the model might try to map too much audio to too little text.
# Let's take the first 12 seconds.
duration_seconds = 12
num_samples_to_keep = int(duration_seconds * sr)

clipped_audio = audio[:num_samples_to_keep]
sf.write(out_path, clipped_audio, sr)

print(f"Original audio was {len(audio)/sr:.2f} seconds.")
print(f"Clipped audio to {duration_seconds} seconds and saved to {out_path}.")
