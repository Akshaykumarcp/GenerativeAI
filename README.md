# TODO
- add table of content
- make sub topic accessible over links
- add point wise
  
## Applications of GANs 
- text-to-image translation
  - Creating Art
  - style transfer
  - photo colorization
- image-to-image translation
  - Image Super-Resolution
  - Creating Art
  - transforming photos from summer to winter or day to night

GANs is successful ability to model high-dimensional data, handle missing data, and the capacity of GANs to provide multi-modal outputs or multiple plausible answers.

## GANS
| Variant  | Paper | Year | Description
| ------------- | ------------- | ------------- | ------------- |
| Generative Adversarial Networks (GAN)  | [link](https://arxiv.org/abs/1406.2661)  | 10 Jun 2014  | The two models, the generator and discriminator, are trained together. The generator generates a batch of samples, and these, along with real examples from the domain, are provided to the discriminator and classified as real or fake. The discriminator is then updated to get better at discriminating real and fake samples in the next round, and importantly, the generator is updated based on how well, or not, the generated samples fooled the discriminator. |
| Conditional Generative Adversarial Nets  | [link](https://arxiv.org/abs/1411.1784) |  6 Nov 2014  | The generative model can be trained to generate new examples from the input domain, where the input, the random vector from the latent space, is provided with (conditioned by) some additional input. |
| NIPS 2016 Tutorial: Generative Adversarial Networks  | [link](https://arxiv.org/abs/1701.00160) |  31 Dec 2016 | - |

## Prompt Engineering
- Resources
  - [OpenAI](https://platform.openai.com/docs/guides/prompt-engineering)
  - [OpenAI suggests](https://cookbook.openai.com/articles/related_resources)
  - [promptingguide.ai](https://www.promptingguide.ai/)
    
| Technique  | Short Notes | References |
| ------------- | ------------- | ------------- |
| Meta Prompt  |  **What?** <br> * Improvise prompt <br> **Why?** <br> * Prompt is an iterative and time consuming <br> **When?** <br> * Optimize prompt <br> * When you feel exhausted while optimizing prompt for a specific task <br> **How?** <br> * Use LLM + prompt for optimizing an specific prompt (ex: by satisfying user query) | [1](https://noahgoodman.substack.com/p/meta-prompt-a-simple-self-improving), [2](https://www.linkedin.com/pulse/meta-prompt-engineering-steve-ball/), [3](https://www.reddit.com/r/ChatGPT/comments/11b08ug/meta_prompt_engineering_chatgpt_creates_amazing/) |

## Fine-tune 

- Instruction fine tune ([paper](https://arxiv.org/abs/2308.10792))

  - Dataset

    - LLM generated dataset
      - [Self-Instruct](https://arxiv.org/abs/2212.10560)
        -    Example dataset: [Alpaca](https://github.com/gururise/AlpacaDataCleaned), was used to fine tune [LLAMA1](https://arxiv.org/abs/2302.13971)
      - [Backtranslation](https://arxiv.org/abs/2308.06259)
    - Human generated dataset
      - [LIMA](https://arxiv.org/abs/2305.11206)
      - [Open Assistant](https://huggingface.co/datasets/OpenAssistant/oasst1)
      - [Natural Instructions](https://arxiv.org/abs/2104.08773)
      - [P3 (Public Pool of Prompts)](https://arxiv.org/abs/2110.08207)
      - [Flan 2021](https://arxiv.org/abs/2301.13688)
      - [others](https://github.com/Lightning-AI/lit-gpt/blob/main/tutorials/prepare_dataset.md) # TODO extract all from this list
  - Libraries for fine tuning
    - [lit](https://github.com/Lightning-AI/lit-gpt)
  - References:
    - [1](https://sebastianraschka.com/blog/2023/optimizing-LLMs-dataset-perspective.html), [2](https://lightning.ai/pages/community/finetuning-falcon-efficiently/) # TODO


## Must Papers
| Title  | Link | Read Status | References |
| ------------- | ------------- | ------------- | ------------- |
| Attention is all you need  | [link](https://arxiv.org/abs/1706.03762)  | Yes | [1](https://magazine.sebastianraschka.com/p/understanding-and-coding-self-attention) |
| Scaling Laws for Neural Language Models | [link](https://arxiv.org/abs/2001.08361)  | No | 
| Instruction Tuning With GPT-4  | [link](https://arxiv.org/pdf/2304.03277.pdf)  | No | 
| AlpaGasus: Training A Better Alpaca with Fewer Data  | [link](https://arxiv.org/abs/2307.08701)  | No | 
| Instruction Tuning for Large Language Models: A Survey  | [link](https://arxiv.org/abs/2308.10792)  | No | 
| Training language models to follow instructions with human feedback (OpenAI's InstructGPT)   | [link](https://arxiv.org/abs/2203.02155)  | No | 


## RLHF 
- References: [1](https://huyenchip.com/2023/05/02/rlhf.html), [2](https://magazine.sebastianraschka.com/p/llm-training-rlhf-and-its-alternatives)

## Large Language Models (SLM)
- [Spreadsheet](https://docs.google.com/spreadsheets/d/1aW4r4WqXZ_RcR9GTpCz7Lwyc2fZP1kpk/edit?usp=sharing&ouid=115305872048583868247&rtpof=true&sd=true) (LLM sheet)

## Small Language Models (SLM)
- [Spreadsheet](https://docs.google.com/spreadsheets/d/1aW4r4WqXZ_RcR9GTpCz7Lwyc2fZP1kpk/edit?usp=sharing&ouid=115305872048583868247&rtpof=true&sd=true) (SLM sheet)
