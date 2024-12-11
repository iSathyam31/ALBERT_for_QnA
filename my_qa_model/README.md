---
library_name: transformers
license: apache-2.0
base_model: albert/albert-base-v2
tags:
- generated_from_trainer
model-index:
- name: my_qa_model
  results: []
---

<!-- This model card has been generated automatically according to the information the Trainer had access to. You
should probably proofread and complete it, then remove this comment. -->

# my_qa_model

This model is a fine-tuned version of [albert/albert-base-v2](https://huggingface.co/albert/albert-base-v2) on an unknown dataset.
It achieves the following results on the evaluation set:
- Loss: 0.9606

## Model description

More information needed

## Intended uses & limitations

More information needed

## Training and evaluation data

More information needed

## Training procedure

### Training hyperparameters

The following hyperparameters were used during training:
- learning_rate: 2e-05
- train_batch_size: 8
- eval_batch_size: 8
- seed: 42
- gradient_accumulation_steps: 2
- total_train_batch_size: 16
- optimizer: Use OptimizerNames.ADAMW_TORCH with betas=(0.9,0.999) and epsilon=1e-08 and optimizer_args=No additional optimizer arguments
- lr_scheduler_type: linear
- num_epochs: 3

### Training results

| Training Loss | Epoch | Step | Validation Loss |
|:-------------:|:-----:|:----:|:---------------:|
| No log        | 1.0   | 250  | 1.0387          |
| 1.1144        | 2.0   | 500  | 0.9152          |
| 1.1144        | 3.0   | 750  | 0.9606          |


### Framework versions

- Transformers 4.47.0
- Pytorch 2.5.1+cu124
- Datasets 3.2.0
- Tokenizers 0.21.0
