# dv_SFT_dataset
Dataset to use for sft learning
Consists of 3000 safety datasets and 5000 usability datasets

# usability datasets
- 'Nemotron-Post-Training-Dataset-v2(5000)' : Multilingual post-training corpus for instruction-following, reasoning, code, math; permissively licensed for improving open models.
- https://huggingface.co/datasets/nvidia/Aegis-AI-Content-Safety-Dataset-2.0

# safety datasets
- `FalseReject(1500)': Ambiguous context, harmless content, but prompts that may seem harmful
- `Aegis(900)`: Unsafe prompt–safe response pairs across safety categories; filtered and stratified by violated category.)
- `Jailbreak(600)`: Adversarial prompts crafted to bypass safety; text-only formats used and stratified by policy.
