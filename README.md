# LLMs-See-Further
Aim to push the boundaries of LLM’s capabilities by enabling them to handle tasks requiring broader context due to their limited ”memory” of past inform

### 1. Implementation of Self-Extend in [LLM Maybe LongLM: Self-Extend LLM Context Window](https://arxiv.org/pdf/2401.01325.pdf). 


**1.1 Overview**

The innate capacity of LLMs to manage lengthy contexts without fine-tuning is evoked by this work. The use of Large Language Models (LLMs) on lengthy input sequences for inference may be restricted due to the training sequence's limited length. 

Building bi-level attention information—the group level and the neighbor level—is the fundamental concept. Since the new model computes the two levels using the self-attention of the original model, no training is necessary.


### 1.2. How to Use SelfExtend

#### _1.2.1 Setup_

For current Llama Implementation, the python packages used are:
```bash
transformers==4.38.2
flash_attn==2.5.6 
```


#### Installation

Clone the repository to your machine and copy your modeling files into the cloned repo directory.

#### _1.2.2 Run_
```python
import SelfExtend

# Load your model, e.g., loaded_model = AutoModelForCausalLM.from_pretrained(model_path) 

# group size, neighbor window. 

SelfExtend.apply(loaded_model, group_size, window_size, enable_flash_attention=False)

# Inference, e.g., loaded_model.generate(...)

```
enable_flash_attention=False by default, you may set enable_flash_attention=True, if the model is loaed with FlashAttention enabled. 

We use passkeyretrieval as an example to show how to use self-extend. You may check [example.py](./example.py):

```bash
python example.py

```
### 1.3 How to choose the group_size and neighbor_window

The ideas that follow are grounded in the implementation:

- **2\~64** are reasonable for group_size using Llama-2 as the basic model; **512\~1536** are possible for neighbor_window. However, in many situations, smaller neighbor windows and greater group sizes are equally beneficial. 

 - When selecting group_size and neighbor_window, the basic approach is to make sure the length of the input sequence is within the maximum extended window size (for llama-2, this would be (4096 - neighbor_window) * group_size + neighbor_window). 

- It could be possible to attempt the least group size [determined by G * (L- w_n) + w_n] for a series of length L first, and then see whether a bigger group works better.


#### _1.3.1 SelfExtend on 'Needle in a Haystack'_
<p align="center">
<img width="600" src="./Results/2d.jpg">
<p align="center">
<img width="600" src="./Results/3d.jpg">

#### _1.3.2 Emperical Rule:_
 SelfExtend is not overly sensitive to hyperparameter selection. One could use a representative task to find proper hyperparameters. Or direcly follow our empirical inequality: $(\frac{1}{2} \sim \frac{2}{3}) \times L > W + \frac{N-W}{G}$

### 1.4 Possible issues unrelated to Self-Extend:
- Gemma-7b has to be loaded in bfloat16. But Gemma-2b still works well with float16.
- If using transformers 4.36, the default attention used by Llama is `LlamaSpdaAttention` rather than `LlamaSpdaAttention`. Be careful about this and make sure you replace the forward method with the correct class.

