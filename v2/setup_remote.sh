#!/bin/bash
set -e
sudo apt-get -qq update && sudo apt-get -qq install -y rsync tmux > /dev/null
python3 -m pip install -q -r ~/v2/requirements.txt
python3 -c "from huggingface_hub import hf_hub_download; print(hf_hub_download('braindecode/eegpt-pretrained','model.safetensors'))"
python3 -c "import torch, braindecode, mne; print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), 'braindecode', braindecode.__version__)"
mkdir -p ~/tuab
