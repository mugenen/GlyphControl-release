# GlyphControl: Glyph Conditional Control for Visual Text Generation

<a href='https://arxiv.org/pdf/2305.18259'><img src='https://img.shields.io/badge/Arxiv-2305.18259-red'>
<a href='https://github.com/AIGText/GlyphControl-release'><img src='https://img.shields.io/badge/Code-GlyphControl-yellow'>
<a href='https://huggingface.co/spaces/AIGText/GlyphControl'><img src='https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-GlyphControl-blue'></a> 

## :star2:	Highlights

* We propose a glyph-conditional text-to-image generation model named **GlyphControl** for visual text generation, which outperforms DeepFloyd IF and Stable Diffusion in terms of OCR accuracy and CLIP score while saving the number of parameters by more than 3×.

* We introduce a visual text generation benchmark named **LAION-Glyph** by filtering the LAION-2B-en and selecting the images with rich visual text content by using the modern OCR system. We conduct experiments on three different dataset scales: LAION-Glyph-100K,LAION-Glyph-1M, and LAION-Glyph-10M.

* We report **flexible** and **customized** visual text generation results. We empirically show that the users can control the content, locations, and sizes of generated visual text through the interface of glyph instructions.


## :hammer_and_wrench: Installation
Clone this repo: 
```
git clone https://github.com/AIGText/GlyphControl-release.git
cd GlyphControl-release
```

Install required python packages
```
conda create -n GlyphControl python=3.9
conda activate GlyphControl
pip install -r requirements.txt

Or

conda env create -f environment_simple.yaml
conda activate GlyphControl
```

<!-- ## :hammer_and_wrench: Inference -->




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