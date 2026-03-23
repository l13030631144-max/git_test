---
library_name: peft
license: other
base_model: /mnt/d/python_project/vllm_learn/Qwen2.5-0.5B-Instruct
tags:
- base_model:adapter:/mnt/d/python_project/vllm_learn/Qwen2.5-0.5B-Instruct
- llama-factory
- lora
- transformers
pipeline_tag: text-generation
model-index:
- name: train_2026-03-23-13-27-14
  results: []
---

<!-- This model card has been generated automatically according to the information the Trainer had access to. You
should probably proofread and complete it, then remove this comment. -->

# train_2026-03-23-13-27-14

This model is a fine-tuned version of [/mnt/d/python_project/vllm_learn/Qwen2.5-0.5B-Instruct](https://huggingface.co//mnt/d/python_project/vllm_learn/Qwen2.5-0.5B-Instruct) on the agent_qa dataset.
It achieves the following results on the evaluation set:
- Loss: 1.4690
- Num Input Tokens Seen: 3607264

## Model description

More information needed

## Intended uses & limitations

More information needed

## Training and evaluation data

More information needed

## Training procedure

### Training hyperparameters

The following hyperparameters were used during training:
- learning_rate: 0.0001
- train_batch_size: 2
- eval_batch_size: 2
- seed: 42
- gradient_accumulation_steps: 8
- total_train_batch_size: 16
- optimizer: Use OptimizerNames.ADAMW_TORCH with betas=(0.9,0.999) and epsilon=1e-08 and optimizer_args=No additional optimizer arguments
- lr_scheduler_type: cosine
- num_epochs: 3.0
- mixed_precision_training: Native AMP

### Training results



### Framework versions

- PEFT 0.18.1
- Transformers 5.2.0
- Pytorch 2.10.0+cu128
- Datasets 4.0.0
- Tokenizers 0.22.2