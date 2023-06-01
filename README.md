# GlyphControl: Glyph Conditional Control for Visual Text Generation

<a href='https://arxiv.org/pdf/2305.18259'><img src='https://img.shields.io/badge/Arxiv-2305.18259-red'>
<a href='https://github.com/AIGText/GlyphControl-release'><img src='https://img.shields.io/badge/Code-GlyphControl-yellow'>
<a href='https://huggingface.co/spaces/AIGText/GlyphControl'><img src='https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-GlyphControl-blue'></a> 

<div align="center">

<img src="readme_files/teaser_6.png" width="90%">

</div>


## :star2:	Highlights

* We propose a glyph-conditional text-to-image generation model named **GlyphControl** for visual text generation, which outperforms DeepFloyd IF and Stable Diffusion in terms of OCR accuracy and CLIP score while saving the number of parameters by more than 3×.

* We introduce a visual text generation benchmark named **LAION-Glyph** by filtering the LAION-2B-en and selecting the images with rich visual text content by using the modern OCR system. We conduct experiments on three different dataset scales: LAION-Glyph-100K,LAION-Glyph-1M, and LAION-Glyph-10M.

* We report **flexible** and **customized** visual text generation results. We empirically show that the users can control the content, locations, and sizes of generated visual text through the interface of glyph instructions.

<div align="center">

<img src="readme_files/architecture.png" width="80%">

</div>

## :floppy_disk: Test Benchmark

* **SimpleBench**: A simple text prompt benchmark following the [Character-aware Paper](https://arxiv.org/abs/2212.10562). The format of prompts remains the same:  _`A sign that says "\<word>".'_
* **CreativeBench**: A creative text prompt benchmark adapted from [GlyphDraw](https://arxiv.org/abs/2303.17870). We adopt diverse English-version prompts in the original benchmark and replace the words inside quotes. As an example, the prompt may look like: _`Little panda holding a sign that says "\<word>".'_ or _'A photographer wears a t-shirt with the word "\<word>." printed on it.'_

(The prompts are listed in the ```text_prompts``` folder)

Following [Character-aware Paper](https://arxiv.org/abs/2212.10562), we collect a pool of single-word candidates from Wikipedia. These words are
then categorized into **four** buckets based on their frequencies: ${Bucket}^{1k}_{top}$, ${Bucket}_{1k}^{10k}$, Bucket^100k^~10~, and Bucket<sub>100k</sub><sup style="margin-left:-23px">plus</sup>. Each bucket contains words with frequencies in the respective range. To form input
prompts, we randomly select **100** words from each bucket and insert them into the above
templates. We generate **four** images for each word during the evaluation process.

## :floppy_disk: Quantitative Results

Method | #Params |Training Dataset  | $\bf{Acc}$ $(\%)$ $\uparrow$ | $\bf{\hat{Acc}}$ $(\%)$ $\uparrow$ |$\bf{LD}\downarrow$ | CLIP Score $\uparrow$ 
:--------- | :--------- | :--------| :---------: | :---------: | :---------: | :---------: |
Stable Diffusion v2.0 | 865M | LAION 1.2B  | $0/0$ | $3/2$ | $4.25/5.01$  | $31.6/33.8$ 
DeepFloyd (IF-I-M) | 2.1B | LAION 1.2B  | $0.3/0.1$ | $18/11$ |  $2.44/3.86$ | $32.8/34.3$  
DeepFloyd (IF-I-L)  | 2.6B | LAION 1.2B  | $0.3/0.7$ | $26/17$ |  $1.97/3.37$ | $33.1/34.9$ 
DeepFloyd (IF-I-XL)  | 6.0B | LAION 1.2B  | $0.6/1$ | $33/21$  | $1.63/3.09$ | $33.5/35.2$
GlyphControl | 1.3B | LAION-Glyph-100K  | $30/19$  |  $37/24$ | $1.77/2.58$ | $33.7/36.2$
GlyphControl | 1.3B |  LAION-Glyph-1M  | $40/26$ |  $45/30$ | $1.59/2.47$ | $33.4/36.0$ 
GlyphControl| 1.3B | LAION-Glyph-10M  | $\bf{42}/\bf{28}$ |  $\bf{48}/\bf{34}$ | $\bf{1.43}/\bf{2.40}$ | $\bf{33.9}/\bf{36.2}$



## :hammer_and_wrench: Installation
Clone this repo: 
```
git clone https://github.com/AIGText/GlyphControl-release.git
cd GlyphControl-release
```

Install required Python packages
```
conda create -n GlyphControl python=3.9
conda activate GlyphControl
pip install -r requirements.txt

Or

conda env create -f environment_simple.yaml
conda activate GlyphControl
```

Althoguh you could run our codes on CPU device,  we recommend you to use CUDA device for faster inference. The recommended CUDA setting is **cuda11.3**.


## :floppy_disk: Available Checkpoints

Download the checkpoints from our [hugging face space](https://huggingface.co/spaces/AIGText/GlyphControl/tree/main/checkpoints) and put the corresponding checkpoint files into the ```checkpoints``` folder. 

We provide **four** types of checkpoints. The relevant information is shown below.

Checkpoint File | Training Dataset  | Trainig Epochs| $\bf{Acc} (\%)\uparrow$ | $\bf{\hat{Acc}} (\%) \uparrow$ |$\bf{LD}\downarrow$ | CLIP Score $\uparrow$ 
:--------- | :--------- | :--------:| :---------: | :---------: | :---------: | :---------: |
laion10M_epoch_6_model_wo_ema.ckpt | LAION-Glyph-10M  | 6 | $\bf{42}/\bf{28}$ |  $\bf{48}/\bf{34}$ | $\bf{1.43}/\bf{2.40}$ | $\bf{33.9}/\bf{36.2}$ | 
textcaps5K_epoch_10_model_wo_ema.ckpt | TextCaps 5K  | 10 | $58/30$  | $64/34$ | $1.01/2.40$ | $33.8/35.1$
textcaps5K_epoch_20_model_wo_ema.ckpt | TextCaps 5K  | 20 | $57/32$  | $66/38$ | $0.97/2.26$ | $34.2/35.5$
textcaps5K_epoch_40_model_wo_ema.ckpt | TextCaps 5K  | 40 | $\bf{71}/\bf{41}$ |  $\bf{77}/\bf{46}$ | $\bf{0.55}/\bf{1.67}$ | $\bf{34.2}/\bf{35.8}$ | 


## :firecracker: Inference
To run inference code locally, you need specify the glyph instructions first in the file ```glyph_instructions.yaml```.

And then execute the code like this:
```
python inference.py --cfg configs/config.yaml --ckpt checkpoints/laion10M_epoch_6_model_wo_ema.ckpt --save_path generated_images --glyph_instructions glyph_instructions.yaml --prompt <Prompt> --num_samples 4
```

If you do not want to generate visual text, you could remove the "--glyph_instructions" parameter in the command.


## :firecracker: Demo (Recommend)
As an easier way to conduct trials on our models, you could test through a demo.

After downloading the checkpoints, execute the code:
```
python app.py
```
Then you could generate visual text through a local demo interface. 

Or you can directly try our **demo** in our **hugging face** space [GlyphControl](https://huggingface.co/spaces/AIGText/GlyphControl).

## :love_letter: Acknowledgement

**Dataset**: 
We sincerely thank the open-source large image-text dataset [LAION-2B-en](https://laion.ai/blog/laion-5b/) and corresponding aesthetic score prediction codes [LAION-Aesthetics_Predictor V2](https://github.com/christophschuhmann/improved-aesthetic-predictor). As for OCR detection, thanks for the open-source tool [PP-OCRv3](https://github.com/PaddlePaddle/PaddleOCR/blob/release/2.6/doc/doc_en/ppocr_introduction_en.md#pp-ocrv3).

**Methodolgy and Demo**:
Our method is based on the powerful controllable image generation method [ControlNet](https://github.com/lllyasviel/ControlNet). Thanks to their open-source codes. As for demo, we use the [ControlNet demo](https://huggingface.co/spaces/hysts/ControlNet) as reference.

**Comparison Methods in the paper**: 
Thanks to the open-source diffusion codes or demos: [DALL-E 2](https://web.augloop-tools.officeppe.com/playground/), [Stable Diffusion 2.0](https://github.com/Stability-AI/StableDiffusion), [Stable Diffusion XL](https://dreamstudio.ai/generate), [DeepFloyd](https://github.com/deep-floyd/IF).



## :envelope: Contact

For help or issues about the github codes or huggingface demo of GlyphControl, please email Yukang Yang (yyk19@tsinghua.org.cn), Dongnan Gui (gdn2001@mail.ustc.edu.cn), and Yuhui Yuan (yuhui.yuan@microsoft.com) or submit a GitHub issue.


## :herb: Citation
If you find this code useful in your research, please consider citing:
```
@misc{yang2023glyphcontrol,
      title={GlyphControl: Glyph Conditional Control for Visual Text Generation}, 
      author={Yukang Yang and Dongnan Gui and Yuhui Yuan and Haisong Ding and Han Hu and Kai Chen},
      year={2023},
      eprint={2305.18259},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```