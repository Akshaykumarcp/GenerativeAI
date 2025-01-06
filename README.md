# TODO
- add table of content
- make sub topic accessible over links
- add point wise

## Table of content 

- GANS
  - [Applications of GANs](#applications-of-gans)
  - [Papers](gans)

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
| ReAct |  **What?** <br> * Reason and Action behaviour in prompt <br> **Why?** <br> * Reason only or Action Only doesn't perform well <br> **When?** <br> * CoT or other technique is not accurate <br> <br> **How?** <br> * Use specific prompt | [1](https://blog.research.google/2022/11/react-synergizing-reasoning-and-acting.html), [2](https://arxiv.org/pdf/2210.03629.pdf) |
| Meta Prompt  |  **What?** <br> * Improvise prompt <br> **Why?** <br> * Prompt is an iterative and time consuming <br> **When?** <br> * Optimize prompt <br> * When you feel exhausted while optimizing prompt for a specific task <br> **How?** <br> * Use LLM + prompt for optimizing an specific prompt (ex: by satisfying user query) | [1](https://noahgoodman.substack.com/p/meta-prompt-a-simple-self-improving), [2](https://www.linkedin.com/pulse/meta-prompt-engineering-steve-ball/), [3](https://www.reddit.com/r/ChatGPT/comments/11b08ug/meta_prompt_engineering_chatgpt_creates_amazing/) |

## Fine-tune 

- Instruction fine tune ([paper](https://arxiv.org/abs/2308.10792))
- Benefits:
  - Better performance. Example prompts acts as internal knowledge of model.
  - Reduce inference cost
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

- Fine Tune with Distillation
  - Example: [1](https://github.com/tatsu-lab/stanford_alpaca)
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
| Training language models to follow instructions with human feedback (OpenAI's InstructGPT)   | [link](https://arxiv.org/abs/2203.02155)  | Yes | [1](https://github.com/openai/following-instructions-human-feedback) |
| Precise Zero-Shot Dense Retrieval without Relevance Labels (HyDE)   | [link](https://arxiv.org/abs/2212.10496)  | No | 
| QLoRA: Efficient Finetuning of Quantized LLMs | [link](https://arxiv.org/abs/2305.14314)  | No |  

## Generative AI services on:
- AWS
  - [Amazon bedrock](https://aws.amazon.com/bedrock/)
    - Foundation model as a service
  - [Amazon Sagemaker](https://aws.amazon.com/sagemaker/), [Amazon Sagemaker Jumpstart](https://aws.amazon.com/sagemaker/jumpstart/)
    - Build own models
  - Training infra
  
## RLHF 
- References: [1](https://huyenchip.com/2023/05/02/rlhf.html), [2](https://magazine.sebastianraschka.com/p/llm-training-rlhf-and-its-alternatives)

## Large Language Models (SLM)
- [Spreadsheet](https://docs.google.com/spreadsheets/d/1aW4r4WqXZ_RcR9GTpCz7Lwyc2fZP1kpk/edit?usp=sharing&ouid=115305872048583868247&rtpof=true&sd=true) (LLM sheet)

## Small Language Models (SLM)
- [Spreadsheet](https://docs.google.com/spreadsheets/d/1aW4r4WqXZ_RcR9GTpCz7Lwyc2fZP1kpk/edit?usp=sharing&ouid=115305872048583868247&rtpof=true&sd=true) (SLM sheet)

## LLM Aspects
- Reliability
  - Inconsistent Outputs
    - Mitigations:
      - [OpenAI techniques](https://cookbook.openai.com/articles/techniques_to_improve_reliability)
      - [OpenAI has added seed and other param for model]()
      - [Build best software engineering best practices around it to handle]()
  - Ambigous output format
    - [Langchain output parser](https://python.langchain.com/docs/modules/model_io/output_parsers/)
- Prompt 
  - Prompt Evaluation
    - Examine few shot prompt engineering
      - Make sure model understand few shot examples
        - test query with similar examples
        - test with complex queries
        - Ask for more examples
        - Ask for exaplanations about few shot examples
  - Prompt Versioning
    - Git
  - Prompt Optimization
    - Experiment with various prompt engineering techniques
- Cost & Latency
  - Cost: Based on use case, compute input and output tokens generation and calculate cost using model pricing.
  - Prompt Vs Fine Tuning
    - 3 factors:
      - Data availability
        - How many examples of prompt needed for fine tune?
          - Depends on model and use case
        - With fewer examples, easy and quickly can get started with prompt
      - Performance
      - Cost
  - Latency: Depends on input and output tokens
    - Input tokens: will be able to process parallely
    - Output tokens: Model will generate token sequentially. More tokens, more time
- Security


## Tools/Softwares for LLM development

- Application Development framework
  - Langchain
    - https://python.langchain.com/en/latest/index.html
  - LLAMAIndex
    - https://github.com/jerryjliu/llama_index
  - Vector Database
    - Pinecone
      - https://www.pinecone.io/
    - Chroma
      - https://www.trychroma.com/
- LLM Experiment tracking
  - weights and biases
    - https://wandb.ai/site/monitoring
  - Truelens
    - https://github.com/truera/trulens/
- Input/Output Safegaurd
  - [LLAMA gaurd](https://ai.meta.com/research/publications/llama-guard-llm-based-input-output-safeguard-for-human-ai-conversations/)
   
## Mixture of experts 
  - Ref
    - [Hugging face](https://huggingface.co/blog/moe)
