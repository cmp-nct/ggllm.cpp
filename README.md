ggllm.cpp is a ggml-backed tool to run quantized Falcon 7B and 40B Models on CPU and GPU

For growing examples and help check the new Wiki:   
https://github.com/cmp-nct/ggllm.cpp/wiki  

**Features that differentiate from llama.cpp for now:**
- Support for Falcon 7B and 40B models (inference, quantization and perplexity tool)
- Fully automated CUDA-GPU offloading based on available and total VRAM
- Run any Falcon Model at up to 16k context without losing sanity
- Current Falcon inference speed on consumer GPU: up to 54+ tokens/sec for 7B and 18-25 tokens/sec for 40B 3-6 bit, roughly 38/sec and 16/sec at at 1000 tokens generated
- Supports running Falcon 40B on a single 4090/3090 (24tk/sec, 15tk/sec), even on a 3080 with a bit of quality sacrifice
- Finetune auto-detection and integrated syntax support (Just load OpenAssistant 7/40 add `-ins` for a chat or `-enc -p "Question"` and optional -sys "System prompt")
- Higher efficiency in VRAM usage when using batched processing (more layers being offloaded)
- 16 bit cuBLAs support (takes half the VRAM for those operations)
- Improved loading screen and visualization
- New tokenizer with regex emulation and BPE merge support

- Optimized RAM and VRAM calculation with batch processing support
- More command line selective features (like disabling GPUs, system prompt, stopwords)

  
**What is missing/being worked on:**
- priority: performance
- web frontend example
- Full GPU offloading of Falcon
- Optimized quantization versions for Falcon
- A new instruct mode
- Large context support (4k-64k in the work)


**Old model support**  
If you use GGML type models (file versions 1-4) you need to place tokenizer.json into the model directory ! (example: https://huggingface.co/OpenAssistant/falcon-40b-sft-mix-1226/blob/main/tokenizer.json)  
If you use updated model binaries they are file version 10+ and called "GGCC", those do not need the load and convert that json file  

**How to just run it?**
1) In most cases you will want to choose a good instruct model, currently the best tunes are from OpenAssist.  
2) Falcon 40B is great even at Q2_K (2 bit) quantization, very good multilingual and reasoning quality.
3) After downloading (and/or converting/quantizing) your model you launch falcon_main with `-enc -p "Your question"` or with `-ins` for multiple questions
4) From there on you can dive into more options, there is a lot to change and optimize.

**The Bloke features fine tuned weights in ggcc v10 with various quantization options:**  
https://huggingface.co/TheBloke/falcon-40b-sft-mix-1226-GGML (OpenAssistant 40B)
https://huggingface.co/TheBloke/falcon-40b-instruct-GGML  
https://huggingface.co/TheBloke/WizardLM-Uncensored-Falcon-40B-GGML  
https://huggingface.co/TheBloke/falcon-7b-instruct-GGML  
https://huggingface.co/TheBloke/WizardLM-Uncensored-Falcon-7B-GGML  
  
**The official HF models are here:**  
https://huggingface.co/tiiuae/falcon-40b/  
https://huggingface.co/tiiuae/falcon-7b/  
https://huggingface.co/tiiuae/falcon-40b-instruct  
https://huggingface.co/tiiuae/falcon-7b-instruct  

**OpenAssistant here:**  
https://huggingface.co/OpenAssistant  
https://huggingface.co/OpenAssistant/falcon-7b-sft-mix-2000  
https://huggingface.co/OpenAssistant/falcon-40b-sft-mix-1226  
_The sft-mix variants appear more capable than the top variants._  
_Download the 7B or 40B Falcon version, use falcon_convert.py (latest version) in 32 bit mode, then falcon_quantize to convert it to ggcc-v10_  

**OpenBuddy**  
https://huggingface.co/OpenBuddy  
_update ggllm.cpp, download the HF directory into openbuddy-7b, then `python falcon_convert.py openbuddy-7b openbuddy-7b 1`, then `falcon_quantize.exe openbuddy-7b/ggml.... openbuddy-7b/q5_1 q5_1 8`_  

**Conversion of HF models and quantization:**
1) use falcon_convert.py to produce a GGML v1 binary from HF - not recommended to be used directly
2) use examples/falcon_quantize to convert these into memory aligned GGCC v10 binaries of your choice including mmap support from there on  
_The Falcon 7B model features tensor sizes which are not yet supported by K-type quantizers - use the traditional quantization for those_

**Conversion of Falcon 180B:**  
The latest falcon_convert.py script supports "safetensors", you will need to upgrade your python transformers libs:  
```
pip install --upgrade transformers
pip install --upgrade accelerate
```
  
**Status/Bugs:**  
- nothing major

**Windows application binary download**
It's always recommended to compile ggllm.cpp fresh yourself so you benefit from the latest features.  
However, in regular intervals full binary compilations are generated: https://github.com/cmp-nct/ggllm.cpp/releases/  
The "master" files contain the executables.
  
**How to compile ggllm.cpp:**
1) Recommended with cmake: (change the CUBLAS flag to 0 to disable CUDA requirements and support)
```
git clone https://github.com/cmp-nct/ggllm.cpp
cd ggllm.cpp
rm -rf build; mkdir build; cd build
# if you do not have cuda in path:
export PATH="/usr/local/cuda/bin:$PATH"
# in case of problems, this sometimes helped
#export CPATH="/usr/local/cuda/targets/x86_64-linux/include:"
#export LD_LIBRARY_PATH="/usr/local/cuda/lib64:"
cmake -DLLAMA_CUBLAS=1 -DCUDAToolkit_ROOT=/usr/local/cuda/ ..  
cmake --build . --config Release
# find the binaries in ./bin
# falcon_main, falcon_quantize, falcon_perplexity
```
2) Building with make (fallback):
```
export LLAMA_CUBLAS=1;
# if you do not have "nvcc" in your path:
# export PATH="/usr/local/cuda/bin:$PATH"
make falcon_main falcon_quantize falcon_perplexity
```

**Windows and Demos**
_Note: those tutorials are before the latest performance patches_
Video tutorial for Windows compilation without WSL:  
https://www.youtube.com/watch?v=BALw669Qeyw     
Another demo of Falcon 40B at 5 bit quantization:  
https://www.youtube.com/watch?v=YuTMFL1dKgQ&ab_channel=CmpNct   
The speed can be seen at 35 tokens/sec start gradually lowering over context - that has been solved in the meantime

3) Installing on WSL (Windows Subsystem for Linux)
```
# Use --no-mmap in WSL OR copy the model into a native directory (not /mnt/) or it will get stuck loading (thanks @nauful)
#Choose a current distro:
wsl.exe --list --online
wsl --install -d distro
# cmake 3.16 is required and the cuda toolset
# If you run an old distro you can upgrade (like apt update; apt upgrade; apt full-upgrade; pico /etc/apt/sources.list/; apt update; apt upgrade; apt full-upgrade; apt autoremove; lsb_release -a); then wsl --shutdown and restart it
# install cuda WSL toolkit
wget https://developer.download.nvidia.com/compute/cuda/repos/wsl-ubuntu/x86_64/cuda-keyring_1.0-1_all.deb
dpkg -i cuda-keyring_1.0-1_all.deb
apt-get update; apt-get -y install cuda
# you might need to add it to your path:
export LD_LIBRARY_PATH="/usr/local/cuda/lib64:$LD_LIBRARY_PATH"
export PATH="/usr/local/cuda/bin:$PATH"
# now start with a fresh cmake and all should work 
```

**CUDA Optimizing inference speed**
- Thread count will be optimal between 1 and 8. Start with `-t 2` 
- For huge prompts n_batch can speed up processing 10-20 times but additional VRAM of 500-1700 MB is required. That's `-b 512` 
- Multi GPU systems can benefit from single GPU processing when the model is small enough. That's `--override-max-gpu 1`  
- Multi GPU systems with different GPUs benefit from custom tensor splitting to load one GPU heavier. To load the 2nd GPU stronger: `--tensor-split 1,3` `-mg 1`
- Need to squeeze a model into VRAM but 1-2 layers don't fit ? Try `--gpu-reserve-mb-main 1` to reduce reserved VRAM to 1 MB, you can use negative numbers to force VRAM swapping
- Wish to reduce VRAM usage and offload less layers? Use `-ngl 10` to only load 10 layers
- Want to dive into details ? Use `--debug-timings <1,2,3>` to get detailed statistics on performance of each operation, how and where it was performed and it's total impact

   
**Inference speed**  
Only some tensors are GPU supported currently and only mul_mat operation supported
Some of the below examples require two GPUs to run at the given speed, the settings were tailored for one environment and a different GPU/CPU/DDR setup might require adaptions  

Below examples are a bit outdated, models are faster now  

**Falcon 40B 6 bit K-type quantization:**
```
falcon_main.exe -m Q:\models\falcon-40b-instruct\q6_k -n 512 -n 32 --debug-timings 0 -b 1 --ignore-eos -p "I am" # -ts 2,1
...
falcon_print_timings:        load time = 11554.93 ms
falcon_print_timings:      sample time =     7.54 ms /    32 runs   (    0.24 ms per token,  4244.59 tokens per second)
falcon_print_timings:        eval time =  1968.34 ms /    33 runs   (   59.65 ms per token,    16.77 tokens per second)
falcon_print_timings:       total time =  1980.28 ms
```

**Falcon 40B 4 bit K-type quantization:**
```
falcon_main.exe -m Q:\models\falcon-40b\q4_k -n 512 -n 128 --debug-timings 0 -b 1 --ignore-eos -p "I am" # -ts 2,1 # --override-max-gpu 1 --gpu-reserve-mb-main -500
...
falcon_print_timings:        load time =  8749.56 ms
falcon_print_timings:      sample time =    29.47 ms /   128 runs   (    0.23 ms per token,  4342.81 tokens per second)
falcon_print_timings:        eval time =  7046.11 ms /   129 runs   (   54.62 ms per token,    18.31 tokens per second)
falcon_print_timings:       total time =  7095.81 ms
```


**Falcon 7B 8 bit quantization:**
```
falcon_main.exe -m Q:\models\falcon-7b-instruct\q8_0 -n 512 -n 32 --debug-timings 0 -b 1 --ignore-eos --override-max-gpu 1 -p "I am"
...
falcon_print_timings:        load time =  2539.21 ms
falcon_print_timings:      sample time =     7.65 ms /    32 runs   (    0.24 ms per token,  4181.91 tokens per second)
falcon_print_timings:        eval time =   758.21 ms /    33 runs   (   22.98 ms per token,    43.52 tokens per second)
falcon_print_timings:       total time =   770.52 ms
```

**Falcon 7B 4 bit quantization (large generation):**
```
falcon_main.exe -t 2 -m Q:\models\falcon-7b\q4_1 -n 512 --debug-timings 0 -b 1 --ignore-eos --override-max-gpu 1 -p "I am"
...
falcon_print_timings:        load time =  2442.76 ms
falcon_print_timings:      sample time =   118.56 ms /   512 runs   (    0.23 ms per token,  4318.34 tokens per second)
falcon_print_timings:        eval time = 16719.48 ms /   769 runs   (   21.74 ms per token,    45.99 tokens per second)
falcon_print_timings:       total time = 16930.51 ms
```

**Falcon 180B on 3090/4090 combination (partial offload):**  
```
falcon_print_timings:        load time = 68642.10 ms
falcon_print_timings:      sample time =    17.99 ms /    70 runs   (    0.26 ms per token,  3891.92 tokens per second)
falcon_print_timings: batch eval time =  2731.63 ms /     9 tokens (  303.51 ms per token,     3.29 tokens per second)
falcon_print_timings:        eval time = 70280.08 ms /    69 runs   ( 1018.55 ms per token,     0.98 tokens per second)
```


CUDA sidenote:  
1) try to use less threads than you have physical processor cores 

