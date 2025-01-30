# MoFO
Code for MoFO: Momentum-Filtered Optimizer for Mitigating Forgetting in LLM Fine-Tuning

## How to use MoFO in your code

Install torch (>=1.8.0) and run the following commands
```python
import algorithms_MoFO

optimizer = algorithms.AdamW_MoFO(
        model = model, lr=learning_rate, weight_decay=weight_decay,fraction=MoFO_fraction)
```

Hyperparameter: `MoFO_fraction` determines the fraction of weights updated in each iteration. For example, `MoFO_fraction=0.15` means 15% parameter with the highest momentum will be updated in each iteration. We recommend setting `MoFO_fraction` between 5% and 20%.

**Note**: Currently, our current implementation of MoFO **does not support CPU offload in DeepSpeed**. Please turn off offload when using DeepSpeed.

## Experiments
We provide some examples of MoFO experiments. We use 4xA800-80GB (for experiments on MetaMathQA) and 2xA800-80GB GPUs (for experiments on  PMC-LLaMA-
Instructions) to run the experiments below. 

We use the [open-instruct](https://github.com/allenai/open-instruct) codebase.

### Setup
```bash
conda env create -n llm1 python==3.10
conda activate llm1

pip install -r requirements_llm1.txt
```


### Model
You can download [Llama-2-7B](https://huggingface.co/meta-llama/Llama-2-7b-hf) and [Llama-2-7B-chat](https://huggingface.co/meta-llama/Llama-2-7b-chat-hf) from Huggingface

### Data
The validation set of datasets are in the `./open_instruct/data` files.

For the training sets of datasets, you can download from:

MetaMathQA:  https://drive.google.com/file/d/1T9Z9PI_C0_B5EBu7KS4oEsnK67hYXqwh/view?usp=drive_link

PMC-LLaMA-Instructions:  https://drive.google.com/file/d/1gw2UlrVNUrOleMZf8JaR1OkRYnmPQA95/view?usp=drive_link


### Run

You can fine-tunine Llama-2-7B on the MetaMathQA dataset by running the following code
```bash
conda activate llm1

./open-instruct/scripts/finetune_metamathqa.sh
```

To be more specific:
* in the shell file,  you will see the code for full fine-tuning the LLM:
```bash
accelerate launch \
    ...
    ./open-instruct/finetune.py \
    ....
```
* To enable MoFO in the fine-tuning, you can add the following option`--use_AdamW_MoFO`, `--MoFO_fraction 0.15`, (`MoFO_fraction=0.15` means 15% parameter with the highest momentum will be updated in each iteration.):
```bash
accelerate launch \
    ...
    ./open_instruct/finetune.py \
    ...
    --use_AdamW_MoFO \
    --MoFO_fraction 0.15 \
    ....
```

### Evaluation
We use [lm-eval](https://github.com/EleutherAI/lm-evaluation-harness) and './openinstruct/eval' of [open-instruct](https://github.com/allenai/open-instruct) codebase to evaluate the fine-tuning performance and forgetting.

#### Prepare dataset for evaluation
You can use the following script to download all the evaluation data in open-instruct database:

```bash
./open-instruct/scripts/prepare_eval_data.sh
```

You can run the following commands to eval `MMLU`, `CommenSense Reasoning`, `HumanEval`, `GSM8K`, (and `MedQ`, `ifeval` by using the commented-out sections)

```bash
./open-instruct/scripts/eval/downstream_benckmark.sh
```
Note that you need to fill `path_of_fine-tuned_ckpt`, `path_where_you_want_to_save_evaluation`, and `path_of_save_eval_data` in the shell.

**NOTE**: We find that due to the mixed precision of fine-tuning, there may be a mismatch in the loss at the 1e-4 level between two repeated fine-tunings, and potentially a 0.5-point discrepancy in the benchmark evaluation.



### Run for Llama-2-7b-chat

setup: 
```bash
conda env create -n llm2 python==3.11
conda activate llm2

pip install -r requirements_llm2.txt
```

run:
You can fine-tunine Llama-2-7B-chat on the MetaMathQA dataset by using the commented-out sections of `./open-instruct/scripts/finetune_metamathqa.sh` with llm2.


You can fine-tunine Llama-2-7B-chat on the PMC-LLaMA-Instructions dataset by using the following commands:
```bash
conda activate llm2

./open-instruct/scripts/finetune_PMC.sh
```
* To enable MoFO in the fine-tuning, you can add the following option`--use_AdamW_MoFO`, `--MoFO_fraction 0.15`, (`MoFO_fraction=0.15` means 15% parameter with the highest momentum will be updated in each iteration.):


You can also evaluate the fine-tuned model by the commands of the `Evaluation` section.

## Citations

If you find this code helpful, please cite our paper in the following format.

```
@article{chen2024mofo,
  title={MoFO: Momentum-Filtered Optimizer for Mitigating Forgetting in LLM Fine-Tuning},
  author={Chen, Yupeng and Wang, Senmiao and Lin, Zhihang and Qin, Zeyu and Zhang, Yushun and Ding, Tian and Sun, Ruoyu},
  journal={arXiv preprint arXiv:2407.20999},
  year={2024}
}
```
