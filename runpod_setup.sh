#!/bin/bash
set -e

cd /workspace
git clone https://github.com/ai4bharat/IndicF5.git
cd IndicF5

# Install requirements
pip install -r requirements.txt
pip install git+https://github.com/ai4bharat/IndicF5.git
pip install hydra-core --upgrade

# Authenticate HuggingFace
huggingface-cli login --token YOUR_HF_TOKEN --add-to-git-credential

# Extract the uploaded dataset
tar -xzvf /workspace/dataset_export.tar.gz

# Create the optimized training configuration
cat << 'EOF' > f5_tts/configs/F5TTS_Base_train.yaml
hydra:
  run:
    dir: ckpts/${model.name}_${model.mel_spec.mel_spec_type}_${model.tokenizer}_${datasets.name}/${now:%Y-%m-%d}/${now:%H-%M-%S}

datasets:
  name: /workspace/IndicF5/custom_dataset_pinyin
  batch_size_per_gpu: 38400
  batch_size_type: frame
  max_samples: 64
  num_workers: 16

optim:
  epochs: 15
  learning_rate: 7.5e-5
  num_warmup_updates: 20000
  grad_accumulation_steps: 1
  max_grad_norm: 1.0
  bnb_optimizer: False

model:
  name: F5TTS_Base
  tokenizer: pinyin
  tokenizer_path: None
  arch:
    dim: 1024
    depth: 22
    heads: 16
    ff_mult: 2
    text_dim: 512
    conv_layers: 4
  mel_spec:
    target_sample_rate: 24000
    n_mel_channels: 100
    hop_length: 256
    win_length: 1024
    n_fft: 1024
    mel_spec_type: vocos
  vocoder:
    is_local: False
    local_path: None

ckpts:
  logger: wandb
  save_per_updates: 50000
  last_per_steps: 5000
  save_dir: ckpts/${model.name}_${model.mel_spec.mel_spec_type}_${model.tokenizer}_${datasets.name}
EOF

# Fast hack for dataset path bug
ln -sf /workspace/IndicF5/custom_dataset_pinyin /workspace/IndicF5/custom_dataset_pinyin_pinyin

# Default accelerate
mkdir -p /root/.cache/huggingface/accelerate
accelerate config default

echo "RunPod Setup Complete!"
