#!/bin/bash
source .env &&
python extract.py &&
python transform.py &&
python validate.py &&
tune download meta-llama/Llama-3.2-3B-Instruct --output-dir ./llama-3.2-3b-instruct --ignore-patterns "original/consolidated*" --hf-token "${HF_TOKEN}" &&
tune run lora_finetune_single_device --config 3B_lora_single_device.yaml &&
sh postprocess.sh
