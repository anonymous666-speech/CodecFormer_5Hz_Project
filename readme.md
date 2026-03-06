# U-Codec: Ultra Low Frame-rate Neural Speech Codec for Fast High-fidelity Speech Generation

[![arXiv](https://img.shields.io/badge/arXiv-2505.16718-brightgreen.svg?style=flat-square)](https://www.arxiv.org/abs/2510.16718)
[![githubio](https://img.shields.io/badge/GitHub.io-Demo_Page-blue?logo=Github&style=flat-square)](https://yangxusheng-yxs.github.io/U-Codec/)
[![GitHub](https://img.shields.io/badge/Github-Code_Release-pink?logo=Github&style=flat-square)](https://github.com/YangXusheng-yxs/CodecFormer_5Hz)
[![HuggingFace](https://img.shields.io/badge/HugginigFace-Stable_Release-blue?style=flat-square)](https://huggingface.co/shaunxsyang/U-Codec)


### News
This paper is currently under review. We have released the checkpoint of U-Codec (5Hz), which can be directly used for inference.


### To do list
- Provide the full training code for the U-Codec framework.

- Release the public code of the TTS models built on top of U-Codec.

If you are interested in U-Codec, feel free to contact us!
## Overview

We propose **U-Codec**, an **U**ltra low frame-rate neural speech **Codec** that achieves high-fidelity reconstruction and fast generation via an extremely frame-rate at 5Hz (5 frames per second). 
Extreme compression at 5Hz typically leads to severe intelligibility and spectral detail loss, we overcome this by integrating a Transformer-based inter-frame long-term dependency module and systematically optimizing residual vector quantization (RVQ) depth and codebook size. 
Moreover, we apply U-Codec into a large language model (LLM)-based auto-regressive TTS model, which leverages global and local hierarchical architecture to effectively capture dependencies across multi-layer tokens. 

The overview of U-Codec as following picture shows.
![The overview of UniAudio](fig/fig2.png)



## How to inference U-Codec
We provide an example to demonstrate how to run U-Codec (5Hz) for audio tokenization and reconstruction.
```
CodecFormer_5Hz/
├── tools/
│   └── tokenizer/
│       └── soundstream/
│           ├── AudioTokenizer_HY.py        # document
│           ├── models/
│           │   └── hy_tokenize.py          # including TokenizerGANWrapper
│           ├── abs_tokenizer.py
│           └── common.py
│       └── hytokenize/
            └── modules
            └── quantization
```


### Environment Setup
First, create a Python environment following a similar setup to [project page](https://github.com/yangdongchao/UniAudio).
```
conda create -n ucodec python=3.8
conda init
source ~/.bashrc
conda activate ucodec
```
Then:
```
cd U-Codec
bash requirements.sh
```
### Run Inference
If you need pretrained weights, please download them on the [Checkpoint](https://huggingface.co/shaunxsyang/U-Codec).

We provide an example script AudioTokenizer_UCodec.py for tokenizing audio into discrete codes and reconstructing audio from the codes.

#### Part 1: reconstruct speech from orignial speech
```
import torch
import torchaudio
from tools.tokenizer.soundstream.AudioTokenizer_HY import HY_Tokenizer

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# initial Tokenizer
tokenizer = HY_Tokenizer(device=device)

# input path
wav_path = ".LibriSpeech/test-clean/8230/279154/8230-279154-0026.wav"

# encode + decode
codes = tokenizer.tokenize(wav_path)
print(f"Token shape: {codes.shape}")  # e.g., (n_q * frames,)

wav_recon = tokenizer.detokenize(codes)
print(f"Reconstructed wav: {wav_recon.shape}")

# save 
torchaudio.save("reconstructed.wav", wav_recon.unsqueeze(0), sample_rate=16000)
print("Saved reconstructed.wav")


```
#### Part 2: Generate discrete codes
```
import torch
from tools.tokenizer.soundstream.AudioTokenizer_HY import HY_Tokenizer

tokenizer = HY_Tokenizer(device=torch.device('cuda:0'))

audio_path = ".LibriSpeech/test-clean/8230/279154/8230-279154-0026.wav"

discrete_code = tokenizer.tokenize(audio_path)
print(f"Discrete token shape: {discrete_code.shape}")
print(discrete_code[:128])  # 打印前128个token

```


#### Part 3: Reconstruction using codes

```
import torch
from tools.tokenizer.soundstream.AudioTokenizer_HY import HY_Tokenizer
import torchaudio

tokenizer = HY_Tokenizer(device=torch.device('cuda:0'))

# 假设已经有 discrete_code
code = torch.load("example_code.pt")  # 或直接使用上面生成的 discrete_code

wav_recon = tokenizer.detokenize(code)
print(f"Decoded wav shape: {wav_recon.shape}")

torchaudio.save("decode_from_code.wav", wav_recon.unsqueeze(0), sample_rate=16000)
print("Saved decode_from_code.wav")

```


#### Part 4: Directly inference
```
cd tools/tokenizer/soundstream
python AudioTokenizer_HY.py
```

You can directly use the released U-Codec 5Hz checkpoint for inference. More examples (e.g., TTS pipeline integration) will be released soon.


### Citation
If you find this code useful in your research, please cite our work and give us a star
```bib
@inproceedings{U-Codec,
  title     = {U-Codec: Ultra Low Frame-rate Neural Speech Codec for Fast High-fidelity Speech Generation},
  author    = {Xusheng Yang, Long Zhou, Wenfu Wang, Kai Hu, Shulin Feng, Chenxing Li, Meng Yu, Dong Yu, Yuexian Zou},
  booktitle = {arXiv},
  year      = {2025}
}
```

### Contact us
If you have any problem about the our code, please contact Xusheng (yangxs@stu.pku.edu.cn). 

### License
You can use the code under MIT license.
