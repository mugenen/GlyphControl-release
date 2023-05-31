# GlyphControl: Glyph Conditional Control for Visual Text Generation

<a href='https://arxiv.org/pdf/2305.18259.pdf'><img src='https://img.shields.io/badge/Arxiv-2305.18259-red'>
<a href='https://github.com/AIGText/GlyphControl-release'><img src='https://img.shields.io/badge/Code-GlyphControl-yellow'>
<a href='https://huggingface.co/spaces/AIGText/GlyphControl'><img src='https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-GlyphControl-blue'></a> 

## :star2:	Highlights

* We propose a glyph-conditional text-to-image generation model named **GlyphControl** for visual text generation, which outperforms DeepFloyd IF and Stable Diffusion in terms of OCR accuracy and CLIP score while saving the number of parameters by more than 3×.

* We introduce a visual text generation benchmark named **LAION-Glyph** by filtering the LAION-2B-en and selecting the images with rich visual text content by using the modern OCR system. We conduct experiments on three different dataset scales: LAION-Glyph-100K,LAION-Glyph-1M, and LAION-Glyph-10M.

* We report **flexible** and **customized** visual text generation results. We empirically show that
the users can control the content, locations, and sizes of generated visual text through the
interface of glyph instructions.
