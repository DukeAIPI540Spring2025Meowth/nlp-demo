#!/bin/bash
# if llama.cpp directory doesn't exist, clone it
if [ ! -d "llama.cpp" ]; then
    git clone https://github.com/ggerganov/llama.cpp &&
    cd llama.cpp && 
    cmake -B build &&
    cmake --build build --config Release &&
    cd ..
fi
python prepare_for_export.py &&
python llama.cpp/convert_hf_to_gguf.py ./conversion_dir --outtype f16 &&
llama.cpp/build/bin/llama-quantize ./conversion_dir/Conversion_Dir-3.2B-F16.gguf ./conversion_dir/model-q5_k_m.gguf q5_k_m &&
python upload.py
