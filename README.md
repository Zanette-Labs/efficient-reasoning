# Training Large Language Models to Reason Efficiently

This is the codebase for our paper ["Training Large Language Models to Reason Efficiently"](https://arxiv.org/abs/2502.04463).
The codebase has been tested on GH200 GPUs with Python 3.10.15 and CUDA 12.6. 
Other environments might require different tweaks in installation of Flash Attention or vLLM. 

## Trained Checkpoints

Trained checkpoints can be found at this ["link"](https://huggingface.co/collections/Zanette-Labs/efficient-reasoning-67f75497d50c1316c04f9332)

## Installation

```
conda create -n efficient_reasoning python=3.10.15
conda activate efficient_reasoning
cd utils/latex2sympy
pip install -e .
cd ../../
pip install -e .
```

## Dataset

Download the dataset used in the paper using:

```
huggingface-cli download daman1209arora/compression_dataset --repo-type dataset --local-dir datasets/compression_dataset
```

This dataset is a random split created using easily parsed problems from the MATH, cn k12, AIME, AoPS and Olympiad subsets Numina Math dataset.

## Design
Our codebase is adapted using the [OpenRLHF](https://github.com/OpenRLHF/OpenRLHF) library.
For minimal changes to the codebase, we launch a remote reward server defined in `reward_server/math_server.py`
which is then passed to the trainer defined in OpenRLHF.


## Usage

For an illustrative example, we provide scripts to run on a slurm cluster:

1. 1.5B with 4 GH200 GPUs on 1 node. `run_rloo_1.5B.sh`
2. 7B with 8 GH200 GPUs on 2 nodes. `run_rloo_7B.sh`

To train the `deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B` model, set `WANDB_KEY` and `ALPHA` in run_rloo_1.5B.sh and run the following command:

```
sbatch run_rloo_1.5B.sh
```

## Evaluation

To evaluate a model, use the script provided in the main directory:

```
python evaluate_model.py \
    --model_path='scale:1.5B_alpha:0.1/' \
    --dataset=openai/gsm8k \
    --scale=1.5B
```

## Citation

If you find this code repository useful, please cite us!

```
@misc{arora2025traininglanguagemodelsreason,
    title={Training Language Models to Reason Efficiently}, 
    author={Daman Arora and Andrea Zanette},
    year={2025},
    eprint={2502.04463},
    archivePrefix={arXiv},
    primaryClass={cs.LG},
    url={https://arxiv.org/abs/2502.04463}, 
}
```
