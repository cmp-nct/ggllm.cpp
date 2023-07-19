#include "falcon_common.h"

#include <cassert>
#include <iostream>
#include <cstring>
#include <fstream>
#include <string>
#include <iterator>
#include <algorithm>
#include <sstream>
#include <unordered_set>
#include <regex>

#if defined(__APPLE__) && defined(__MACH__)
#include <sys/types.h>
#include <sys/sysctl.h>
#endif

#if defined(_WIN32)
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#include <fcntl.h>
#include <io.h>
#else
#include <sys/ioctl.h>
#include <unistd.h>
#include <wchar.h>
#endif

int32_t get_num_physical_cores() {
#ifdef __linux__
    // enumerate the set of thread siblings, num entries is num cores
    std::unordered_set<std::string> siblings;
    for (uint32_t cpu=0; cpu < UINT32_MAX; ++cpu) {
        std::ifstream thread_siblings("/sys/devices/system/cpu"
            + std::to_string(cpu) + "/topology/thread_siblings");
        if (!thread_siblings.is_open()) {
            break; // no more cpus
        }
        std::string line;
        if (std::getline(thread_siblings, line)) {
            siblings.insert(line);
        }
    }
    if (siblings.size() > 0) {
        return static_cast<int32_t>(siblings.size());
    }
#elif defined(__APPLE__) && defined(__MACH__)
    int32_t num_physical_cores;
    size_t len = sizeof(num_physical_cores);
    int result = sysctlbyname("hw.perflevel0.physicalcpu", &num_physical_cores, &len, NULL, 0);
    if (result == 0) {
        return num_physical_cores;
    }
    result = sysctlbyname("hw.physicalcpu", &num_physical_cores, &len, NULL, 0);
    if (result == 0) {
        return num_physical_cores;
    }
#elif defined(_WIN32)
    int logical_cores;
    SYSTEM_INFO sysinfo;
    GetSystemInfo(&sysinfo);
    logical_cores = sysinfo.dwNumberOfProcessors;

    DWORD_PTR process_affinity_mask;
    DWORD_PTR system_affinity_mask;
    GetProcessAffinityMask(GetCurrentProcess(), &process_affinity_mask, &system_affinity_mask);

    int physical_cores = 0;
    for (int i = 0; i < sizeof(DWORD_PTR) * 8; i++) {
        if (process_affinity_mask & ((DWORD_PTR)1 << i)) {
            physical_cores++;
        }
    }
    return physical_cores;
#endif
    unsigned int n_threads = std::thread::hardware_concurrency();
    return n_threads > 0 ? (n_threads <= 4 ? n_threads : n_threads / 2) : 4;
}

void process_escapes(std::string& input) {
    std::size_t input_len = input.length();
    std::size_t output_idx = 0;

    for (std::size_t input_idx = 0; input_idx < input_len; ++input_idx) {
        if (input[input_idx] == '\\' && input_idx + 1 < input_len) {
            switch (input[++input_idx]) {
                case 'n':  input[output_idx++] = '\n'; break;
                case 'r':  input[output_idx++] = '\r'; break;
                case 't':  input[output_idx++] = '\t'; break;
                case '\'': input[output_idx++] = '\''; break;
                case '\"': input[output_idx++] = '\"'; break;
                case '\\': input[output_idx++] = '\\'; break;
                default:   input[output_idx++] = '\\';
                           input[output_idx++] = input[input_idx]; break;
            }
        } else {
            input[output_idx++] = input[input_idx];
        }
    }

    input.resize(output_idx);
}

bool gpt_params_parse(int argc, char ** argv, gpt_params & params) {
    bool invalid_param = false;
    bool escape_prompt = false;
    std::string arg;
    gpt_params default_params;
    const std::string arg_prefix = "--";

    #if defined(GGML_USE_CUBLAS)
        ggml_cuda_set_max_gpus(LLAMA_MAX_DEVICES); // default
    #endif
    params.n_threads = get_num_physical_cores();
    // until thread scheduling is improved, these numbers are around the optimal (for huge batch processing increase -t manually)
    if (params.n_threads > 8) params.n_threads = 4;
    if (params.n_threads > 4) params.n_threads = 2;
    params.seed = (int) time(NULL); // initiate a seed - we need one if multiple context used with similar input

    

    for (int i = 1; i < argc; i++) {
        arg = argv[i];
        if (arg.compare(0, arg_prefix.size(), arg_prefix) == 0) {
            std::replace(arg.begin(), arg.end(), '_', '-');
        }

        if (arg == "-s" || arg == "--seed") {
#if defined(GGML_USE_CUBLAS)
            // fprintf(stderr, "WARNING: when using cuBLAS generation results are NOT guaranteed to be reproducible.\n");
#endif
            if (++i >= argc) {
                invalid_param = true;
                break;
            }
            params.seed = std::stoi(argv[i]);
        } else if (arg == "-t" || arg == "--threads") {
            if (++i >= argc) {
                invalid_param = true;
                break;
            }
            params.n_threads = std::stoi(argv[i]);
        } else if (arg == "-p" || arg == "--prompt") {
            if (++i >= argc) {
                invalid_param = true;
                break;
            }
            // params.prompt = argv[i];
            params.prompt += argv[i];
        } else if (arg == "-e") {
            escape_prompt = true;
        } else if (arg == "--prompt-cache") {
            if (++i >= argc) {
                invalid_param = true;
                break;
            }
            params.path_prompt_cache = argv[i];
        } else if (arg == "--prompt-cache-all") {
            params.prompt_cache_all = true;
        } else if (arg == "--prompt-cache-ro") {
            params.prompt_cache_ro = true;
        } else if (arg == "-f" || arg == "--file") {
            if (++i >= argc) {
                invalid_param = true;
                break;
            }
            std::ifstream file(argv[i]);
            if (!file) {
                fprintf(stderr, "error: failed to open file '%s'\n", argv[i]);
                invalid_param = true;
                break;
            }
            std::copy(std::istreambuf_iterator<char>(file), std::istreambuf_iterator<char>(), back_inserter(params.prompt));
            if (params.prompt.back() == '\n') {
                params.prompt.pop_back();
            }
        } else if (arg == "-sysf" || arg == "--system-file") {
            if (++i >= argc) {
                invalid_param = true;
                break;
            }
            std::ifstream file(argv[i]);
            if (!file) {
                fprintf(stderr, "error: failed to open file '%s'\n", argv[i]);
                invalid_param = true;
                break;
            }
            std::copy(std::istreambuf_iterator<char>(file), std::istreambuf_iterator<char>(), back_inserter(params.system_prompt));
            if (params.system_prompt.back() == '\n') {
                params.system_prompt.pop_back();
            }
        } else if (arg == "-n" || arg == "--n-predict") {
            if (++i >= argc) {
                invalid_param = true;
                break;
            }
            params.n_predict = std::stoi(argv[i]);
        } else if (arg == "--top-k") {
            if (++i >= argc) {
                invalid_param = true;
                break;
            }
            params.top_k = std::stoi(argv[i]);
            params.sampling_not_default=true;
        } else if (arg == "-c" || arg == "--ctx-size") {
            if (++i >= argc) {
                invalid_param = true;
                break;
            }
            params.n_ctx = std::stoi(argv[i]);
        } else if (arg == "--memory-f16") {
            params.memory_f16 = true;
            params.kvmem_not_default=true;
        } else if (arg == "--memory-f32") {
            params.memory_f16 = false;
            params.kvmem_not_default=true;
        } else if (arg == "--top-p") {
            if (++i >= argc) {
                invalid_param = true;
                break;
            }
            params.top_p = std::stof(argv[i]);
            params.sampling_not_default=true;
        } else if (arg == "--temp") {
            if (++i >= argc) {
                invalid_param = true;
                break;
            }
            params.temp = std::stof(argv[i]);
            params.sampling_not_default=true;
        } else if (arg == "--tfs") {
            if (++i >= argc) {
                invalid_param = true;
                break;
            }
            params.tfs_z = std::stof(argv[i]);
        } else if (arg == "--typical") {
            if (++i >= argc) {
                invalid_param = true;
                break;
            }
            params.typical_p = std::stof(argv[i]);
            params.sampling_not_default=true;
        } else if (arg == "--repeat-last-n") {
            if (++i >= argc) {
                invalid_param = true;
                break;
            }
            params.repeat_last_n = std::stoi(argv[i]);
        } else if (arg == "--repeat-penalty") {
            if (++i >= argc) {
                invalid_param = true;
                break;
            }
            params.repeat_penalty = std::stof(argv[i]);
        } else if (arg == "--frequency-penalty") {
            if (++i >= argc) {
                invalid_param = true;
                break;
            }
            params.frequency_penalty = std::stof(argv[i]);
        } else if (arg == "--presence-penalty") {
            if (++i >= argc) {
                invalid_param = true;
                break;
            }
            params.presence_penalty = std::stof(argv[i]);
        } else if (arg == "--mirostat") {
            if (++i >= argc) {
                invalid_param = true;
                break;
            }
            params.mirostat = std::stoi(argv[i]);
            params.sampling_not_default=true;
        } else if (arg == "--mirostat-lr") {
            if (++i >= argc) {
                invalid_param = true;
                break;
            }
            params.mirostat_eta = std::stof(argv[i]);
        } else if (arg == "--mirostat-ent") {
            if (++i >= argc) {
                invalid_param = true;
                break;
            }
            params.mirostat_tau = std::stof(argv[i]);
        } else if (arg == "-b" || arg == "--batch-size") {
            if (++i >= argc) {
                invalid_param = true;
                break;
            }
            params.n_batch = std::stoi(argv[i]);
            // params.n_batch = std::min(1024+128, params.n_batch); // appears to work fine with scratch buffer, keep in eye
        } else if (arg == "--keep") {
            if (++i >= argc) {
                invalid_param = true;
                break;
            }
            params.n_keep = std::stoi(argv[i]);
        } else if (arg == "-m" || arg == "--model") {
            if (++i >= argc) {
                invalid_param = true;
                break;
            }
            params.model = argv[i];
        } else if (arg == "-a" || arg == "--alias" || arg == "--finetune" ) {
            if (++i >= argc) {
                invalid_param = true;
                break;
            }
            params.model_alias = argv[i];
            // force a finetune type based on alias
            if (params.model_alias == "wizard") {
                params.finetune_type = FINETUNE_WIZARD;
            } else
            if (params.model_alias == "falcon-ins") {
                params.finetune_type = FINETUNE_FALCONINSTRUCT;
            } else
            if (params.model_alias == "open-assistant") { // the current openassist using special tokens like <|prompter|> and <|assistant|>
                params.finetune_type = FINETUNE_OPENASSISTANT;
            } else
            if (params.model_alias == "open-assistant-v1") { // a older non special tokens finetune using <|prompt|>
                params.finetune_type = FINETUNE_OPENASSIST_V1;
            } else
            if (params.model_alias == "alpaca") {
                params.finetune_type = FINETUNE_ALPACA;
            } else
            if (params.model_alias == "none") {
                params.finetune_type = FINETUNE_NONE;
            } 
        }  else if (arg == "-S" || arg == "--stopwords") {
            if (++i >= argc) {
                invalid_param = true;
                break;
            }
            params.stopwords = argv[i];
        } else if (arg == "-enc" || arg == "-enclose") {
            params.enclose_finetune = true;
        }  else if (arg == "-sys" || arg == "--system") {
            if (++i >= argc) {
                invalid_param = true;
                break;
            }
            params.system_prompt = argv[i];
        } else if (arg == "-sysraw" || arg == "--system-raw") {
            params.sys_prompt_is_raw = true;
        }
        else if (arg == "--lora") {
            if (++i >= argc) {
                invalid_param = true;
                break;
            }
            params.lora_adapter = argv[i];
            params.use_mmap = false;
        } else if (arg == "--lora-base") {
            if (++i >= argc) {
                invalid_param = true;
                break;
            }
            params.lora_base = argv[i];
        } else if (arg == "-i" || arg == "--interactive") {
            params.interactive = true;
        } else if (arg == "--embedding") {
            params.embedding = true;
        } else if (arg == "--interactive-first") {
            params.interactive_first = true;
        } else if (arg == "-ins" || arg == "--instruct") {
            params.instruct = true;
            params.interactive = true;
            params.enclose_finetune = true;
        } else if (arg == "--multiline-input") {
            params.multiline_input = true;
        } else if (arg == "--color") {
            params.use_color = true;
        } else if (arg == "--mlock") {
            params.use_mlock = true;
        } else if (arg == "--gpu-layers" || arg == "-ngl" || arg == "--n-gpu-layers") {
            if (++i >= argc) {
                invalid_param = true;
                break;
            }
            #ifdef LLAMA_SUPPORTS_GPU_OFFLOAD
            params.n_gpu_layers = std::stoi(argv[i]);
            #else
            fprintf(stderr, "warning: not compiled with GPU offload support, --n-gpu-layers option will be ignored\n");
            fprintf(stderr, "warning: see main README.md for information on enabling GPU BLAS support\n");
            #endif
        } else if (arg == "--main-gpu" || arg == "-mg") {
            if (++i >= argc) {
                invalid_param = true;
                break;
            }
            #ifdef GGML_USE_CUBLAS
            params.main_gpu = std::stoi(argv[i]);
            ggml_cuda_set_main_device(params.main_gpu);
            #else
            fprintf(stderr, "warning: falcon.cpp was compiled without cuBLAS. It is not possible to set a main GPU.\n");
            #endif
        } else if (arg == "--override-max-gpu") {
            if (++i >= argc) {
                invalid_param = true;
                break;
            }
            #ifdef GGML_USE_CUBLAS
            params.n_max_gpu = std::stoi(argv[i]);
            ggml_cuda_set_max_gpus(params.n_max_gpu);
            #else
            fprintf(stderr, "warning: falcon.cpp was compiled without cuBLAS. It is not possible limit GPU devices.\n");
            #endif
        } else if (arg == "--gpu-reserve-mb-main") {
            if (++i >= argc) {
                invalid_param = true;
                break;
            }
            #ifdef GGML_USE_CUBLAS
            params.mb_reserve_gpu_main = std::stoi(argv[i]);
            ggml_cuda_set_vram_reserved(((int64_t)params.mb_reserve_gpu_main)*1024*1024);
            #else
            fprintf(stderr, "warning: falcon.cpp was compiled without cuBLAS. VRAM not available.\n");
            #endif
        } /*else if (arg == "--gpu-reserve-mb-other") {
            if (++i >= argc) {
                invalid_param = true;
                break;
            }
            #ifdef GGML_USE_CUBLAS
            params.mb_reserve_gpu_other = std::stoi(argv[i]);
            #else
            fprintf(stderr, "warning: falcon.cpp was compiled without cuBLAS. VRAM not available.\n");
            #endif
        } */else if (arg == "--tensor-split" || arg == "-ts") {
            if (++i >= argc) {
                invalid_param = true;
                break;
            }
            #ifdef GGML_USE_CUBLAS
            std::string arg_next = argv[i];

            // split string by , and /
            const std::regex regex{R"([,/]+)"};
            std::sregex_token_iterator it{arg_next.begin(), arg_next.end(), regex, -1};
            std::vector<std::string> split_arg{it, {}};
            GGML_ASSERT(split_arg.size() <= LLAMA_MAX_DEVICES);
            bool all_zero = true;
            for (size_t i = 0; i < LLAMA_MAX_DEVICES; ++i) {
                if (i < split_arg.size()) {
                    params.tensor_split[i] = std::stof(split_arg[i]);
                    if (params.tensor_split[i] != 0.0f) {
                        all_zero = false;
                    }
                } else {
                    params.tensor_split[i] = 0.0f;
                }
            }
            if (all_zero) {
                fprintf(stderr, "Error: all tensor split proportions are zero\n");
                exit(1);
            }
            ggml_cuda_set_tensor_split_prepare(params.tensor_split, static_cast<int>(split_arg.size()));
            #else
            fprintf(stderr, "warning: falcon.cpp was compiled without cuBLAS. It is not possible to set a tensor split.\n");
            #endif // GGML_USE_CUBLAS
        } else if (arg == "--no-mmap") {
            params.use_mmap = false;
        } else if (arg == "--mtest") {
            params.mem_test = true;
        } else if (arg == "--export") {
            params.export_cgraph = true;
        } else if (arg == "--debug-timings" || arg == "--display-timings" || arg == "-dt") {
            if (++i >= argc) {
                params.debug_timings = 1;
            } else
                params.debug_timings = std::stoi(argv[i]);
        } else if (arg == "--verbose-prompt") {
            params.verbose_prompt = true;
        } else if (arg == "-r" || arg == "--reverse-prompt") {
            if (++i >= argc) {
                invalid_param = true;
                break;
            }
            params.antiprompt.push_back(argv[i]);
        } else if (arg == "--perplexity") {
            params.perplexity = true;
        } else if (arg == "--ignore-eos") {
            params.logit_bias[falcon_token_eos()] = -INFINITY; // todo: make it a proper stopword removal bool
        } else if (arg == "--no-penalize-nl") {
            params.penalize_nl = false;
        } else if (arg == "-l" || arg == "--logit-bias") {
            if (++i >= argc) {
                invalid_param = true;
                break;
            }
            std::stringstream ss(argv[i]);
            falcon_token key;
            char sign;
            std::string value_str;
            try {
                if (ss >> key && ss >> sign && std::getline(ss, value_str) && (sign == '+' || sign == '-')) {
                    params.logit_bias[key] = std::stof(value_str) * ((sign == '-') ? -1.0f : 1.0f);
                } else {
                    throw std::exception();
                }
            } catch (const std::exception &e) {
                (void)e;
                invalid_param = true;
                break;
            }
        } else if (arg == "-h" || arg == "--help") {
            gpt_print_usage(argc, argv, default_params);
            exit(0);
        } else if (arg == "--random-prompt") {
            params.random_prompt = true;
        } else if (arg == "--in-prefix") {
            if (++i >= argc) {
                invalid_param = true;
                break;
            }
            params.input_prefix = argv[i];
        } else if (arg == "--in-suffix") {
            if (++i >= argc) {
                invalid_param = true;
                break;
            }
            params.input_suffix = argv[i];
        } else {
            fprintf(stderr, "error: unknown argument: %s\n", arg.c_str());
            gpt_print_usage(argc, argv, default_params);
            exit(1);
        }
    }
    if (invalid_param) {
        fprintf(stderr, "error: invalid parameter for argument: %s\n", arg.c_str());
        gpt_print_usage(argc, argv, default_params);
        exit(1);
    }
    if (params.prompt_cache_all &&
            (params.interactive || params.interactive_first ||
             params.instruct)) {
        fprintf(stderr, "error: --prompt-cache-all not supported in interactive mode yet\n");
        gpt_print_usage(argc, argv, default_params);
        exit(1);
    }
    if (escape_prompt) {
        process_escapes(params.prompt);
        process_escapes(params.system_prompt);
    }


    bool all_zero = true;
        for (size_t i = 0; i < LLAMA_MAX_DEVICES; ++i) {
            if (params.tensor_split[i] != 0.0f) {
                all_zero = false;
                break;
            }
        }
        if (!all_zero) {
            if (params.tensor_split[params.main_gpu] == 0.0f) {
                fprintf(stderr, "Error: main GPU cannot have a tensor split proportion of zero.\n");
                exit(1);
            }
        }

    return true;
}

void gpt_print_usage(int /*argc*/, char ** argv, const gpt_params & params) {
    fprintf(stderr, "usage: %s [options]\n", argv[0]);
    fprintf(stderr, "\n");
    fprintf(stderr, "options:\n");
    fprintf(stderr, "  -h, --help            show this help message and exit\n");
    fprintf(stderr, "  -i, --interactive, -ins \n");
    fprintf(stderr, "                        run in interactive chat mode\n");
    fprintf(stderr, "  --interactive-first   wait for user input after prompt ingestion\n");
    // fprintf(stderr, "  -ins, --instruct      run in instruction mode (use with Alpaca models)\n");
    fprintf(stderr, "  -a,--alias,--finetune Set model name alias and optionally force fine-tune type (or disable it)\n");
    fprintf(stderr, "                        Finetune options: wizard, falcon-ins, open-assistant, alpaca, none\n");
    fprintf(stderr, "                        Use if finetune autodetection does not or wrongly recognizes your model or filename\n");
    fprintf(stderr, "  -sys, --system  <>    prefix the entire prompt with the system prompt text\n");
    fprintf(stderr, "  -sysraw, --system-raw treat the system prompt raw (do not add syntax)\n");
    // fprintf(stderr, "  --sys_prompt_simple    trust the model to follow the system prompt instead of using evaluated sampling adaption\n");
    fprintf(stderr, "  -enc, --enclose       enclose the prompt in fine-tune optimal syntax\n");
    fprintf(stderr, "                        This automatically chooses the correct syntax to write around your prompt.\n");
    fprintf(stderr, "  --multiline-input     allows you to write or paste multiple lines without ending each in '\\'\n");
    // fprintf(stderr, "  -r PROMPT, --reverse-prompt PROMPT\n");
    // fprintf(stderr, "                        halt generation at PROMPT, return control in interactive mode\n");
    // fprintf(stderr, "                        (can be specified more than once for multiple prompts).\n");
    fprintf(stderr, "  --color               colorise output to distinguish prompt and user input from generations\n");
    fprintf(stderr, "  -s SEED, --seed SEED  RNG seed (default: -1, use random seed for < 0)\n");
    fprintf(stderr, "  -t N, --threads N     number of threads to use during computation (default: %d)\n", params.n_threads);
    fprintf(stderr, "  -p PROMPT, --prompt PROMPT\n");
    fprintf(stderr, "                        prompt to start generation with (default: empty)\n");
    fprintf(stderr, "                        you can use multiple -p to append them, also in combination with -f\n");
    fprintf(stderr, "  -S, --stopwords \",,,\" Add stopwords in addition to the usual end-of-sequence\n");
    fprintf(stderr, "                        comma separated list: -S \"\\n,Hello World,stopword\" - overwrites defaults except eos\n");
    fprintf(stderr, "                        Important: 'is' and ' is' are unique tokens in a stopword. Just as 'Hello' and ' Hello' are distinct\n");
    fprintf(stderr, "  -e                    process escapes sequences in prompt and system message (\\n, \\r, \\t, \\', \\\", \\\\)\n");
    fprintf(stderr, "  --prompt-cache FNAME  file to cache prompt state for faster startup (default: none)\n");
    fprintf(stderr, "  --prompt-cache-all    if specified, saves user input and generations to cache as well.\n");
    fprintf(stderr, "                        not supported with --interactive or other interactive options\n");
    fprintf(stderr, "  --prompt-cache-ro     if specified, uses the prompt cache but does not update it.\n");
    fprintf(stderr, "  --random-prompt       start with a randomized prompt.\n");
    fprintf(stderr, "  --in-prefix STRING    string to prefix user inputs with (default: empty)\n");
    fprintf(stderr, "  --in-suffix STRING    string to suffix after user inputs with (default: empty)\n");
    fprintf(stderr, "  -f FNAME, --file FNAME\n");
    fprintf(stderr, "                        read prompt from a file, optionally -p prompt is prefixed\n");
    fprintf(stderr, "  -sysf FNAME, --system-file FNAME\n");
    fprintf(stderr, "                        read system prompt from a file\n");
    fprintf(stderr, "  -n N, --n-predict N   number of tokens to predict (default: %d, -1 = infinity)\n", params.n_predict);
    fprintf(stderr, "  --top-k N             top-k sampling (default: %d, 0 = disabled)\n", params.top_k);
    fprintf(stderr, "  --top-p N             top-p sampling (default: %.1f, 1.0 = disabled)\n", (double)params.top_p);
    fprintf(stderr, "  --tfs N               tail free sampling, parameter z (default: %.1f, 1.0 = disabled)\n", (double)params.tfs_z);
    fprintf(stderr, "  --typical N           locally typical sampling, parameter p (default: %.1f, 1.0 = disabled)\n", (double)params.typical_p);
    fprintf(stderr, "  --repeat-last-n N     last n tokens to consider for penalize (default: %d, 0 = disabled, -1 = ctx_size)\n", params.repeat_last_n);
    fprintf(stderr, "  --repeat-penalty N    penalize repeat sequence of tokens (default: %.1f, 1.0 = disabled)\n", (double)params.repeat_penalty);
    fprintf(stderr, "  --presence-penalty N  repeat alpha presence penalty (default: %.1f, 0.0 = disabled)\n", (double)params.presence_penalty);
    fprintf(stderr, "  --frequency-penalty N repeat alpha frequency penalty (default: %.1f, 0.0 = disabled)\n", (double)params.frequency_penalty);
    fprintf(stderr, "  --mirostat N          use Mirostat sampling.\n");
    fprintf(stderr, "                        Top K, Nucleus, Tail Free and Locally Typical samplers are ignored if used.\n");
    fprintf(stderr, "                        (default: %d, 0 = disabled, 1 = Mirostat, 2 = Mirostat 2.0)\n", params.mirostat);
    fprintf(stderr, "  --mirostat-lr N       Mirostat learning rate, parameter eta (default: %.1f)\n", (double)params.mirostat_eta);
    fprintf(stderr, "  --mirostat-ent N      Mirostat target entropy, parameter tau (default: %.1f)\n", (double)params.mirostat_tau);
    fprintf(stderr, "  -l TOKEN_ID(+/-)BIAS, --logit-bias TOKEN_ID(+/-)BIAS\n");
    fprintf(stderr, "                        modifies the likelihood of token appearing in the completion,\n");
    fprintf(stderr, "                        i.e. `--logit-bias 15043+1` to increase likelihood of token ' Hello',\n");
    fprintf(stderr, "                        or `--logit-bias 15043-1` to decrease likelihood of token ' Hello'\n");
    fprintf(stderr, "  -c N, --ctx-size N    size of the prompt context (default: %d)\n", params.n_ctx);
    fprintf(stderr, "  --ignore-eos          ignore end of stream token and continue generating (implies --logit-bias 2-inf)\n");
    fprintf(stderr, "  --no-penalize-nl      do not penalize newline token\n");
    fprintf(stderr, "  --memory-f16          force f16 kv mem instead of f32 for memory key+value (default: adaptive)\n");
    fprintf(stderr, "  --memory-f32          force f32 kv mem (default: adaptive)\n");
    fprintf(stderr, "  --temp N              temperature (default: %.1f)\n", (double)params.temp);
    fprintf(stderr, "  -b N, --batch-size N  batch size for prompt processing (default: %d)\n", params.n_batch);
    fprintf(stderr, "  --perplexity          compute perplexity over the prompt\n");
    fprintf(stderr, "  --keep                number of tokens to keep from the initial prompt (default: %d, -1 = all)\n", params.n_keep);
    if (llama_mlock_supported()) {
        fprintf(stderr, "  --mlock               force system to keep model in RAM rather than swapping or compressing\n");
    }
    if (llama_mmap_supported()) {
        fprintf(stderr, "  --no-mmap             Do not memory-map model (slower load but may reduce pageouts if not using mlock)\n");
    }
#ifdef LLAMA_SUPPORTS_GPU_OFFLOAD
    fprintf(stderr, "  -ngl N, --n-gpu-layers N\n");
    fprintf(stderr, "                        number of layers to store in VRAM, default is to fill VRAM \n");
    fprintf(stderr, "  -ts SPLIT --tensor-split SPLIT\n");
    fprintf(stderr, "                        how to split tensors across multiple GPUs, comma-separated list of proportions, e.g. 3,1\n");
    fprintf(stderr, "  -mg i, --main-gpu i   the GPU to use for scratch and small tensors (0 = first)\n" );
    fprintf(stderr, "  --override-max-gpu N\n");
    fprintf(stderr, "                        limits the number of GPUs visible (allows to disable multi/single GPU processing)\n");
    fprintf(stderr, "  --gpu-reserve-mb-main override reserved total VRAM MB (can be negative if your driver supports swapping into RAM) \n");
    //fprintf(stderr, "  --gpu_reserve_mb_other override reserved VRAM MB for other GPUs (for multi GPU systems)\n");
#endif
    fprintf(stderr, "  --mtest               compute maximum memory usage\n");
    fprintf(stderr, "  --export              export the computation graph to 'llama.ggml'\n");
    fprintf(stderr, "  --verbose-prompt      print prompt before generation\n");
    fprintf(stderr, "  -dt, --debug-timings  print GGML_PERF debug output (requires GGML_PERF=1 for timings)\n");
    fprintf(stderr, "                        1 = print first layer, 2 = print first and last layer, 3+ = all layers\n");
    fprintf(stderr, "  --lora FNAME          apply LoRA adapter (implies --no-mmap)\n");
    fprintf(stderr, "  --lora-base FNAME     optional model to use as a base for the layers modified by the LoRA adapter\n");
    fprintf(stderr, "  -m FNAME, --model FNAME\n");
    fprintf(stderr, "                        model path (default: %s)\n", params.model.c_str());
    fprintf(stderr, "\n");
}

std::string gpt_random_prompt(std::mt19937 & rng) {
    const int r = rng() % 10;
    switch (r) {
        case 0: return "So";
        case 1: return "Once upon a time";
        case 2: return "When";
        case 3: return "The";
        case 4: return "After";
        case 5: return "If";
        case 6: return "import";
        case 7: return "He";
        case 8: return "She";
        case 9: return "They";
        default: return "To";
    }

    return "The";
}

// TODO: not great allocating this every time
std::vector<falcon_token> falcon_tokenize(struct falcon_context * ctx, const std::string & text, bool add_bos) {
    // initialize to prompt numer of chars, since n_tokens <= n_prompt_chars
    std::vector<falcon_token> res(text.size() + (int) add_bos);
    const int n = falcon_tokenize(ctx, text.c_str(), res.data(), static_cast<int>(res.size()), add_bos);
    assert(n >= 0);
    res.resize(n);

    return res;
}
struct falcon_context_params falcon_context_params_create(const gpt_params &params)
{
    auto lparams = falcon_context_default_params();

    lparams.n_ctx        = params.n_ctx;
    lparams.n_batch      = params.n_batch;
    lparams.n_gpu_layers = params.n_gpu_layers;
    lparams.main_gpu     = params.main_gpu;
    memcpy(lparams.tensor_split, params.tensor_split, LLAMA_MAX_DEVICES*sizeof(float));
    lparams.seed         = params.seed;
    lparams.f16_kv       = params.memory_f16;
    lparams.use_mmap     = params.use_mmap;
    lparams.use_mlock    = params.use_mlock;
    lparams.logits_all   = params.perplexity;
    lparams.embedding    = params.embedding;

    return lparams;
}

struct falcon_context * falcon_init_from_gpt_params(const gpt_params & params) {
    
    struct falcon_context_params lparams = falcon_context_params_create(params);
    falcon_context * lctx = falcon_init_from_file(params.model.c_str(), lparams);

    if (lctx == NULL) {
        fprintf(stderr, "%s: error: failed to load model '%s'\n", __func__, params.model.c_str());
        return NULL;
    }

    // if (!params.lora_adapter.empty()) {
    //     int err = llama_apply_lora_from_file(lctx,
    //                                          params.lora_adapter.c_str(),
    //                                          params.lora_base.empty() ? NULL : params.lora_base.c_str(),
    //                                          params.n_threads);
    //     if (err != 0) {
    //         fprintf(stderr, "%s: error: failed to apply lora adapter\n", __func__);
    //         return NULL;
    //     }
    // }

    return lctx;
}

void console_init(console_state & con_st) {
#if defined(_WIN32)
    // Windows-specific console initialization
    DWORD dwMode = 0;
    con_st.hConsole = GetStdHandle(STD_OUTPUT_HANDLE);
    if (con_st.hConsole == INVALID_HANDLE_VALUE || !GetConsoleMode(con_st.hConsole, &dwMode)) {
        con_st.hConsole = GetStdHandle(STD_ERROR_HANDLE);
        if (con_st.hConsole != INVALID_HANDLE_VALUE && (!GetConsoleMode(con_st.hConsole, &dwMode))) {
            con_st.hConsole = NULL;
        }
    }
    if (con_st.hConsole) {
        // Enable ANSI colors on Windows 10+
        if (con_st.use_color && !(dwMode & ENABLE_VIRTUAL_TERMINAL_PROCESSING)) {
            SetConsoleMode(con_st.hConsole, dwMode | ENABLE_VIRTUAL_TERMINAL_PROCESSING);
        }
        // Set console output codepage to UTF8
        SetConsoleOutputCP(CP_UTF8);
    }
    HANDLE hConIn = GetStdHandle(STD_INPUT_HANDLE);
    if (hConIn != INVALID_HANDLE_VALUE && GetConsoleMode(hConIn, &dwMode)) {
        // Set console input codepage to UTF16
        _setmode(_fileno(stdin), _O_WTEXT);

        // Turn off ICANON (ENABLE_LINE_INPUT) and ECHO (ENABLE_ECHO_INPUT)
        dwMode &= ~(ENABLE_LINE_INPUT | ENABLE_ECHO_INPUT);
        SetConsoleMode(hConIn, dwMode);
    }
#else
    // POSIX-specific console initialization
    struct termios new_termios;
    tcgetattr(STDIN_FILENO, &con_st.prev_state);
    new_termios = con_st.prev_state;
    new_termios.c_lflag &= ~(ICANON | ECHO);
    new_termios.c_cc[VMIN] = 1;
    new_termios.c_cc[VTIME] = 0;
    tcsetattr(STDIN_FILENO, TCSANOW, &new_termios);

    con_st.tty = fopen("/dev/tty", "w+");
    if (con_st.tty != nullptr) {
        con_st.out = con_st.tty;
    }

    setlocale(LC_ALL, "");
#endif
}

void console_cleanup(console_state & con_st) {
    // Reset console color
    console_set_color(con_st, CONSOLE_COLOR_DEFAULT);

#if !defined(_WIN32)
    if (con_st.tty != nullptr) {
        con_st.out = stdout;
        fclose(con_st.tty);
        con_st.tty = nullptr;
    }
    // Restore the terminal settings on POSIX systems
    tcsetattr(STDIN_FILENO, TCSANOW, &con_st.prev_state);
#endif
}

/* Keep track of current color of output, and emit ANSI code if it changes. */
void console_set_color(console_state & con_st, console_color_t color) {
    if (con_st.use_color && con_st.color != color) {
        fflush(stdout);
        switch(color) {
            case CONSOLE_COLOR_DEFAULT:
                fprintf(con_st.out, ANSI_COLOR_RESET);
                break;
            case CONSOLE_COLOR_PROMPT:
                fprintf(con_st.out, ANSI_COLOR_YELLOW);
                break;
            case CONSOLE_COLOR_USER_INPUT:
                fprintf(con_st.out, ANSI_BOLD ANSI_COLOR_GREEN);
                break;
            case CONSOLE_COLOR_ERROR:
                fprintf(con_st.out, ANSI_BOLD ANSI_COLOR_RED);
                break;
        }
        con_st.color = color;
        fflush(con_st.out);
    }
}

char32_t getchar32() {
#if defined(_WIN32)
    HANDLE hConsole = GetStdHandle(STD_INPUT_HANDLE);
    wchar_t high_surrogate = 0;

    while (true) {
        INPUT_RECORD record;
        DWORD count;
        if (!ReadConsoleInputW(hConsole, &record, 1, &count) || count == 0) {
            return WEOF;
        }

        if (record.EventType == KEY_EVENT && record.Event.KeyEvent.bKeyDown) {
            wchar_t wc = record.Event.KeyEvent.uChar.UnicodeChar;
            if (wc == 0) {
                continue;
            }

            if ((wc >= 0xD800) && (wc <= 0xDBFF)) { // Check if wc is a high surrogate
                high_surrogate = wc;
                continue;
            } else if ((wc >= 0xDC00) && (wc <= 0xDFFF)) { // Check if wc is a low surrogate
                if (high_surrogate != 0) { // Check if we have a high surrogate
                    return ((high_surrogate - 0xD800) << 10) + (wc - 0xDC00) + 0x10000;
                }
            }

            high_surrogate = 0; // Reset the high surrogate
            return static_cast<char32_t>(wc);
        }
    }
#else
    wchar_t wc = getwchar();
    if (static_cast<wint_t>(wc) == WEOF) {
        return WEOF;
    }

#if WCHAR_MAX == 0xFFFF
    if ((wc >= 0xD800) && (wc <= 0xDBFF)) { // Check if wc is a high surrogate
        wchar_t low_surrogate = getwchar();
        if ((low_surrogate >= 0xDC00) && (low_surrogate <= 0xDFFF)) { // Check if the next wchar is a low surrogate
            return (static_cast<char32_t>(wc & 0x03FF) << 10) + (low_surrogate & 0x03FF) + 0x10000;
        }
    }
    if ((wc >= 0xD800) && (wc <= 0xDFFF)) { // Invalid surrogate pair
        return 0xFFFD; // Return the replacement character U+FFFD
    }
#endif

    return static_cast<char32_t>(wc);
#endif
}

void pop_cursor(console_state & con_st) {
#if defined(_WIN32)
    if (con_st.hConsole != NULL) {
        CONSOLE_SCREEN_BUFFER_INFO bufferInfo;
        GetConsoleScreenBufferInfo(con_st.hConsole, &bufferInfo);

        COORD newCursorPosition = bufferInfo.dwCursorPosition;
        if (newCursorPosition.X == 0) {
            newCursorPosition.X = bufferInfo.dwSize.X - 1;
            newCursorPosition.Y -= 1;
        } else {
            newCursorPosition.X -= 1;
        }

        SetConsoleCursorPosition(con_st.hConsole, newCursorPosition);
        return;
    }
#endif
    putc('\b', con_st.out);
}

int estimateWidth(char32_t codepoint) {
#if defined(_WIN32)
    return 1;
#else
    return wcwidth(codepoint);
#endif
}

int put_codepoint(console_state & con_st, const char* utf8_codepoint, size_t length, int expectedWidth) {
#if defined(_WIN32)
    CONSOLE_SCREEN_BUFFER_INFO bufferInfo;
    if (!GetConsoleScreenBufferInfo(con_st.hConsole, &bufferInfo)) {
        // go with the default
        return expectedWidth;
    }
    COORD initialPosition = bufferInfo.dwCursorPosition;
    DWORD nNumberOfChars = (DWORD)length;
    WriteConsole(con_st.hConsole, utf8_codepoint, nNumberOfChars, &nNumberOfChars, NULL);

    CONSOLE_SCREEN_BUFFER_INFO newBufferInfo;
    GetConsoleScreenBufferInfo(con_st.hConsole, &newBufferInfo);

    // Figure out our real position if we're in the last column
    if (utf8_codepoint[0] != 0x09 && initialPosition.X == newBufferInfo.dwSize.X - 1) {
        DWORD nNumberOfChars;
        WriteConsole(con_st.hConsole, &" \b", 2, &nNumberOfChars, NULL);
        GetConsoleScreenBufferInfo(con_st.hConsole, &newBufferInfo);
    }

    int width = newBufferInfo.dwCursorPosition.X - initialPosition.X;
    if (width < 0) {
        width += newBufferInfo.dwSize.X;
    }
    return width;
#else
    // we can trust expectedWidth if we've got one
    if (expectedWidth >= 0 || con_st.tty == nullptr) {
        fwrite(utf8_codepoint, length, 1, con_st.out);
        return expectedWidth;
    }

    fputs("\033[6n", con_st.tty); // Query cursor position
    int x1, x2, y1, y2;
    int results = 0;
    results = fscanf(con_st.tty, "\033[%d;%dR", &y1, &x1);

    fwrite(utf8_codepoint, length, 1, con_st.tty);

    fputs("\033[6n", con_st.tty); // Query cursor position
    results += fscanf(con_st.tty, "\033[%d;%dR", &y2, &x2);

    if (results != 4) {
        return expectedWidth;
    }

    int width = x2 - x1;
    if (width < 0) {
        // Calculate the width considering text wrapping
        struct winsize w;
        ioctl(STDOUT_FILENO, TIOCGWINSZ, &w);
        width += w.ws_col;
    }
    return width;
#endif
}

void replace_last(console_state & con_st, char ch) {
#if defined(_WIN32)
    pop_cursor(con_st);
    put_codepoint(con_st, &ch, 1, 1);
#else
    fprintf(con_st.out, "\b%c", ch);
#endif
}

void append_utf8(char32_t ch, std::string & out) {
    if (ch <= 0x7F) {
        out.push_back(static_cast<unsigned char>(ch));
    } else if (ch <= 0x7FF) {
        out.push_back(static_cast<unsigned char>(0xC0 | ((ch >> 6) & 0x1F)));
        out.push_back(static_cast<unsigned char>(0x80 | (ch & 0x3F)));
    } else if (ch <= 0xFFFF) {
        out.push_back(static_cast<unsigned char>(0xE0 | ((ch >> 12) & 0x0F)));
        out.push_back(static_cast<unsigned char>(0x80 | ((ch >> 6) & 0x3F)));
        out.push_back(static_cast<unsigned char>(0x80 | (ch & 0x3F)));
    } else if (ch <= 0x10FFFF) {
        out.push_back(static_cast<unsigned char>(0xF0 | ((ch >> 18) & 0x07)));
        out.push_back(static_cast<unsigned char>(0x80 | ((ch >> 12) & 0x3F)));
        out.push_back(static_cast<unsigned char>(0x80 | ((ch >> 6) & 0x3F)));
        out.push_back(static_cast<unsigned char>(0x80 | (ch & 0x3F)));
    } else {
        // Invalid Unicode code point
    }
}

// Helper function to remove the last UTF-8 character from a string
void pop_back_utf8_char(std::string & line) {
    if (line.empty()) {
        return;
    }

    size_t pos = line.length() - 1;

    // Find the start of the last UTF-8 character (checking up to 4 bytes back)
    for (size_t i = 0; i < 3 && pos > 0; ++i, --pos) {
        if ((line[pos] & 0xC0) != 0x80) break; // Found the start of the character
    }
    line.erase(pos);
}

bool console_readline(console_state & con_st, std::string & line) {
    console_set_color(con_st, CONSOLE_COLOR_USER_INPUT);
    if (con_st.out != stdout) {
        fflush(stdout);
    }

    line.clear();
    std::vector<int> widths;
    bool is_special_char = false;
    bool end_of_stream = false;

    char32_t input_char;
    while (true) {
        fflush(con_st.out); // Ensure all output is displayed before waiting for input
        input_char = getchar32();

        if (input_char == '\r' || input_char == '\n') {
            break;
        }

        if (input_char == (char32_t) WEOF || input_char == 0x04 /* Ctrl+D*/) {
            end_of_stream = true;
            break;
        }

        if (is_special_char) {
            console_set_color(con_st, CONSOLE_COLOR_USER_INPUT);
            replace_last(con_st, line.back());
            is_special_char = false;
        }

        if (input_char == '\033') { // Escape sequence
            char32_t code = getchar32();
            if (code == '[' || code == 0x1B) {
                // Discard the rest of the escape sequence
                while ((code = getchar32()) != (char32_t) WEOF) {
                    if ((code >= 'A' && code <= 'Z') || (code >= 'a' && code <= 'z') || code == '~') {
                        break;
                    }
                }
            }
        } else if (input_char == 0x08 || input_char == 0x7F) { // Backspace
            if (!widths.empty()) {
                int count;
                do {
                    count = widths.back();
                    widths.pop_back();
                    // Move cursor back, print space, and move cursor back again
                    for (int i = 0; i < count; i++) {
                        replace_last(con_st, ' ');
                        pop_cursor(con_st);
                    }
                    pop_back_utf8_char(line);
                } while (count == 0 && !widths.empty());
            }
        } else {
            int offset = (int)line.length();
            append_utf8(input_char, line);
            int width = put_codepoint(con_st, line.c_str() + offset, line.length() - offset, estimateWidth(input_char));
            if (width < 0) {
                width = 0;
            }
            widths.push_back(width);
        }

        if (!line.empty() && (line.back() == '\\' || line.back() == '/')) {
            console_set_color(con_st, CONSOLE_COLOR_PROMPT);
            replace_last(con_st, line.back());
            is_special_char = true;
        }
    }

    bool has_more = con_st.multiline_input;
    if (is_special_char) {
        replace_last(con_st, ' ');
        pop_cursor(con_st);

        char last = line.back();
        line.pop_back();
        if (last == '\\') {
            line += '\n';
            fputc('\n', con_st.out);
            has_more = !has_more;
        } else {
            // llama will just eat the single space, it won't act as a space
            if (line.length() == 1 && line.back() == ' ') {
                line.clear();
                pop_cursor(con_st);
            }
            has_more = false;
        }
    } else {
        if (end_of_stream) {
            has_more = false;
        } else {
            line += '\n';
            fputc('\n', con_st.out);
        }
    }

    fflush(con_st.out);
    return has_more;
}
