import os
import torch
import numpy as np
import torchaudio
from f5_tts.model import DiT
from f5_tts.infer.utils_infer import load_model, load_vocoder, load_checkpoint, infer_batch_process
from huggingface_hub import hf_hub_download

# USE THE SAME reference audio that the model was fine-tuned against
ref_audio = "custom_prompts/tamil_male_reference_clipped.wav"
ref_text = "அந்தக் கிராமத்துல ஒரு சின்ன பையன் இருந்தான் அவன் பேரு கண்ணன் கண்ணனுக்கு எப்பவும் புதுசு புதுசா ஏதாச்சும் கத்துக்கணும்னு ரொம்ப ஆசை"
gen_text = "அந்தக் கிராமத்துல ஒரு சின்ன பையன் இருந்தான் அவன் பேரு கண்ணன் கண்ணனுக்கு எப்பவும் புதுசு புதுசா ஏதாச்சும் கத்துக்கணும்னு ரொம்ப ஆசை"

print("Downloading Fine-Tuned Model and Vocab from Hugging Face...")
repo_id = "ananthgv-usk/IndicF5-Tamil-Finetuned"
ckpt_path = hf_hub_download(repo_id=repo_id, filename="model_last.pt")
vocab_file = hf_hub_download(repo_id=repo_id, filename="vocab.txt")

device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")

vocoder = load_vocoder(is_local=False)
model_obj = load_model(model_cls=DiT, model_cfg=dict(dim=1024, depth=22, heads=16, ff_mult=2, text_dim=512, conv_layers=4), mel_spec_type="vocos", vocab_file=vocab_file, ode_method="euler", use_ema=True, device=device)
model_obj = load_checkpoint(model_obj, ckpt_path, device, use_ema=True)

audio, sr = torchaudio.load(ref_audio)
print(f"Original audio shape: {audio.shape}, sr: {sr}")

try:
    final_wave, sr, spect = infer_batch_process(
        (audio, sr),
        ref_text,
        [gen_text],
        model_obj,
        vocoder,
        mel_spec_type="vocos",
        device=device,
        fix_duration=21.0  # Same duration that produced the good finetuned_test_4.wav
    )
    print("Success! Final wave shape:", final_wave.shape)
    
    # Peak normalize to fix uneven volume
    wave = np.array(final_wave, dtype=np.float32)
    peak = np.max(np.abs(wave))
    if peak > 0:
        wave = wave / peak * 0.95
    
    os.makedirs("samples", exist_ok=True)
    torchaudio.save("samples/runpod_finetuned_test.wav", torch.tensor(wave).unsqueeze(0), sr)
    print(f"Saved! peak={np.max(np.abs(wave)):.3f}")
except Exception as e:
    import traceback
    traceback.print_exc()
