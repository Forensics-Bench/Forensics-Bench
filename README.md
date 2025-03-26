# Forensics-Bench

<p align="left">
  <!-- <a href="#🚀-quick-start"><b>Quick Start</b></a> | -->
  <a href="https://forensics-bench.github.io/"><b>HomePage</b></a> |
  <a href="https://arxiv.org/abs/2503.15024"><b>arXiv</b></a> |
  <a href="https://huggingface.co/datasets/Forensics-bench/Forensics-bench"><b>Dataset</b></a> |
  <!-- <a href="#🖊️-citation"><b>Citation</b></a> <br> -->
</p>

This repository is the official implementation of [Forensics-Bench](https://arxiv.org/abs/2503.15024). 

> [Forensics-Bench: A Comprehensive Forgery Detection Benchmark Suite for Large Vision Language Models](https://arxiv.org/abs/2503.15024)  
> Jin Wang<sup>\*</sup>, Chenghui Lv<sup>\*</sup>, Xian Li, Shichao Dong, Huadong Li, Kelu Yao, Chao Li, Wenqi Shao, Ping Luo<sup>†</sup>
> <sup>\*</sup> JW and CL contribute equally (primary contact: <a href="mailto:wj0529@connect.hku.hk">wj0529@connect.hku.hk</a>).  
> <sup>†</sup> Ping Luo is correponding author. 

## News 💡

- `2025-03-24`: We release the evaluation code.
- `2025-03-22`: We released the `Forensics-Bench` dataset. 
- `2025-02-27`: Our paper has been accepted to CVPR 2025!

## Introduction
Forensics-Bench is a new forgery detection evaluation benchmark suite to assess LVLMs across massive forgery detection tasks, requiring comprehensive recognition, location and reasoning capabilities on diverse forgeries. Forensics-Bench comprises <b>63,292</b> meticulously curated multi-choice visual questions, covering <b>112</b> unique forgery detection types from <b>5</b> perspectives: forgery semantics, forgery modalities, forgery tasks, forgery types and forgery models. We conduct thorough evaluations on <b>22</b> open-sourced LVLMs and <b>3</b> proprietary models GPT-4o, Gemini 1.5 Pro, and Claude 3.5 Sonnet, highlighting the significant challenges of comprehensive forgery detection posed by Forensics-Bench. We anticipate that Forensics-Bench will motivate the community to advance the frontier of LVLMs, striving for all-around forgery detectors in the era of AIGC.
![overview](assets/FDBENCH2.png)

---

# Quickstart

We use [VLMEevalKit](https://github.com/open-compass/VLMEvalKit) as our evaluation framework. It is a highly user-friendly framework that requires only minimal configuration to get started.

Before running the evaluation scripts, you need to configure the VLMs and correctly set the `model_paths` in `vlmeval/config.py`.

After that, you can use a single script `run.py` to inference and evaluate VLMs on the Forensics-Bench.

## Step 0. Installation & Setup essential keys

**Installation.**

```bash
git clone https://github.com/Forensics-Bench/Forensics-Bench.git
cd Forensics-Bench
pip install -e .
```

**Setup Keys.**

To infer with API models (GPT-4o, Gemini-Pro, etc.) or use LLM APIs as the **judge or choice extractor**, you need to first setup API keys. VLMEvalKit will use an judge **LLM** to extract answer from the output if you set the key, otherwise it uses the **exact matching** mode (find "Yes", "No", "A", "B", "C"... in the output strings). **The exact matching can only be applied to the Yes-or-No tasks and the Multi-choice tasks.**

- You can place the required keys in `$Forensics-Bench/.env` or directly set them as the environment variable. If you choose to create a `.env` file, its content will look like:

  ```bash
  # The .env file, place it under $VLMEvalKit
  # API Keys of Proprietary VLMs
  # QwenVL APIs
  DASHSCOPE_API_KEY=
  # Gemini w. Google Cloud Backends
  GOOGLE_API_KEY=
  # OpenAI API
  OPENAI_API_KEY=
  OPENAI_API_BASE=
  # StepAI API
  STEPAI_API_KEY=
  # REKA API
  REKA_API_KEY=
  # GLMV API
  GLMV_API_KEY=
  # CongRong API
  CW_API_BASE=
  CW_API_KEY=
  # SenseChat-V API
  SENSECHAT_AK=
  SENSECHAT_SK=
  # Hunyuan-Vision API
  HUNYUAN_SECRET_KEY=
  HUNYUAN_SECRET_ID=
  # LMDeploy API
  LMDEPLOY_API_BASE=
  # You can also set a proxy for calling api models during the evaluation stage
  EVAL_PROXY=
  ```

- Fill the blanks with your API keys (if necessary). Those API keys will be automatically loaded when doing the inference and evaluation.

## Step 1. Configuration

**VLM Configuration**: All VLMs are configured in `vlmeval/config.py`. Few legacy VLMs (like MiniGPT-4, LLaVA-v1-7B) requires additional configuration (configuring the code / model_weight root in the config file). During evaluation, you should use the model name specified in `supported_VLM` in `vlmeval/config.py` to select the VLM. Make sure you can successfully infer with the VLM before starting the evaluation with the following command `vlmutil check {MODEL_NAME}`.

Following VLMs require the configuration step:  
**Code Preparation & Installation**: InstructBLIP ([LAVIS](https://github.com/salesforce/LAVIS)), LLaVA & LLaVA-Next & Yi-VL ([LLaVA](https://github.com/haotian-liu/LLaVA)), mPLUG-Owl2 ([mPLUG-Owl2](https://github.com/X-PLUG/mPLUG-Owl/tree/main/mPLUG-Owl2)), DeepSeek-VL ([DeepSeek-VL](https://github.com/deepseek-ai/DeepSeek-VL)).  
**Manual Weight Preparation**: For InstructBLIP, you also need to modify the config files in `vlmeval/vlm/misc` to configure LLM path and ckpt path.

**Dataset Configuration**: Download the dataset [ForensicsBench.tsv](https://huggingface.co/datasets/Forensics-bench/Forensics-bench) from Hugging Face and place it in the `Forensics-Bench/LMUData` directory.

**Transformers Version Recommendation:**

Note that some VLMs may not be able to run under certain transformer versions, we recommend the following settings to evaluate each VLM:

- **Please use** `transformers==4.33.0` **for**: `Qwen series`, `Monkey series`, `InternLM-XComposer Series`, `mPLUG-Owl2`, `VisualGLM`, `MMAlaya`, `InstructBLIP series`.
- **Please use** `transformers==4.37.0` **for**: `LLaVA series`, `ShareGPT4V series`, `LLaVA (XTuner)`, `CogVLM Series`, `Yi-VL Series`, `DeepSeek-VL series`, `InternVL series`.
- **Please use** `transformers==latest` **for**: `LLaVA-Next series`.

## Step 2. Evaluation

We use `run.py` for evaluation. To use the script, you can use `$Forensics-Bench/run.py` or create a soft-link of the script (to use the script anywhere):

**Arguments** 

- `--data (list[str])`: Set the dataset names `ForensicsBench`.
- `--model (list[str])`: Set the VLM names that are supported in VLMEvalKit (defined in `supported_VLM` in `vlmeval/config.py`).
- `--mode (str, default to 'all', choices are ['all', 'infer'])`: When `mode` set to "all", will perform both inference and evaluation; when set to "infer", will only perform the inference.
- `--api-nproc (int, default to 4)`: The number of threads for OpenAI API calling.
- `--work-dir (str, default to '.')`: The directory to save evaluation results.

**Command for Evaluating Forensics-Bench:**  You can run the script with `python`:

```bash
# When running with `python`, only one VLM instance is instantiated, and it might use multiple GPUs (depending on its default behavior).
# That is recommended for evaluating very large VLMs.

# InternVL-Chat-V1-2 on ForensicsBench, Inference and Evalution
python run.py --data ForensicsBench --model InternVL-Chat-V1-2 --verbose
# InternVL-Chat-V1-2 on ForensicsBench, Inference only
python run.py --data ForensicsBench --model InternVL-Chat-V1-2 --verbose --mode infer
```

The evaluation results will be printed as logs, besides. **Result Files** will also be generated in the directory `$YOUR_WORKING_DIRECTORY/{model_name}`. Files ending with `.csv` contain the evaluated metrics.

**Summary Scores:**  After evaluating the models, you can run the following script to view the summary scores.

```bash
python summary_scores.py --filename /path/to/your/csv
```


## 💐 Acknowledgement

We expressed sincerely gratitude for the projects listed following:

- [VLMEvalKit](https://github.com/open-compass/VLMEvalKit) provides useful out-of-box tools and implements many adavanced LVLMs. Thanks for their selfless dedication.

## 🖊️ Citation

If you feel Forensics-Bench useful in your project or research, please kindly use the following BibTeX entry to cite our paper. Thanks!

```
@misc{wang2025forensicsbenchcomprehensiveforgerydetection,
    title={Forensics-Bench: A Comprehensive Forgery Detection Benchmark Suite for Large Vision Language Models}, 
    author={Jin Wang and Chenghui Lv and Xian Li and Shichao Dong and Huadong Li and kelu Yao and Chao Li and Wenqi Shao and Ping Luo},
    year={2025},
    eprint={2503.15024},
    archivePrefix={arXiv},
    primaryClass={cs.CV},
    url={https://arxiv.org/abs/2503.15024}, 
 }
```
