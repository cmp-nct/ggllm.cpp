/**
 * @file falcon_main.cpp
 * @brief Falcon main application
 * https://github.com/cmp-nct/ggllm.cpp
 * MIT licensed, contributions welcome
 */
#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif

#include "falcon_common.h"
#include "libfalcon.h"
#include "build-info.h"

#include <cassert>
#include <cinttypes>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <ctime>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#if defined (__unix__) || (defined (__APPLE__) && defined (__MACH__))
#include <signal.h>
#include <unistd.h>
#elif defined (_WIN32)
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#include <signal.h>
#include <shellapi.h>
#endif

static console_state con_st;
static falcon_context ** g_ctx;

std::unordered_map<std::string, int> special_tokens_ids = {
     {">>TITLE<<", 0},
     {">>ABSTRACT<<", 1},
     {">>INTRODUCTION<<", 2},
     {">>SUMMARY<<", 3},
     {">>COMMENT<<", 4},
     {">>ANSWER<<", 5},
     {">>QUESTION<<", 6},
     {">>DOMAIN<<", 7},
     {">>PREFIX<<", 8},
     {">>SUFFIX<<", 9},
     {">>MIDDLE<<", 10},
     {"<|endoftext|>", 11},
};




static bool is_interacting = false;

#if defined (__unix__) || (defined (__APPLE__) && defined (__MACH__)) || defined (_WIN32)
void sigint_handler(int signo) {
    if (signo == SIGINT) {
        if (!is_interacting) {
            is_interacting=true;
        } else {
            console_cleanup(con_st);
            printf("\n");
            falcon_print_timings(*g_ctx);
            _exit(130);
        }
    }
}
#endif

int main(int argc, char ** argv) {
    #if defined(_WIN32)
    SetConsoleOutputCP(CP_UTF8);
    int wargc;
    wchar_t** wargv = CommandLineToArgvW(GetCommandLineW(), &wargc);
    if (wargv == nullptr)
    {
        fprintf(stderr, "Failed to parse command line\n");
        exit(1);
    }
    // Convert from UTF-16 to UTF-8
    std::vector<char*> utf8argv(wargc);
    for (int i = 0; i < wargc; ++i)
    {
        int size = WideCharToMultiByte(CP_UTF8, 0, wargv[i], -1, nullptr, 0, nullptr, nullptr);
        utf8argv[i] = new char[size];
        WideCharToMultiByte(CP_UTF8, 0, wargv[i], -1, utf8argv[i], size, nullptr, nullptr);
    }
    LocalFree(wargv);
    // Use utf8argv instead of argv
    argv = utf8argv.data();
    #endif
    gpt_params params;

    if (gpt_params_parse(argc, argv, params) == false) {
        return 1;
    }

    // save choice to use color for later
    // (note for later: this is a slightly awkward choice)
    con_st.use_color = params.use_color;
    con_st.multiline_input = params.multiline_input;
    console_init(con_st);
    atexit([]() { console_cleanup(con_st); });

    if (params.perplexity) {
        printf("\n************\n");
        printf("%s: please use the 'perplexity' tool for perplexity calculations\n", __func__);
        printf("************\n\n");

        return 0;
    }

    if (params.embedding) {
        printf("\n************\n");
        printf("%s: please use the 'embedding' tool for embedding calculations\n", __func__);
        printf("************\n\n");

        return 0;
    }

    if (params.n_ctx > 2048) {
        fprintf(stderr, "%s: warning: model does not support context sizes greater than 2048 tokens (%d specified);"
                "expect poor results\n", __func__, params.n_ctx);
    } else if (params.n_ctx < 8) {
        fprintf(stderr, "%s: warning: minimum context size is 8, using minimum size.\n", __func__);
        params.n_ctx = 8;
    }

    fprintf(stderr, "%s: build = %d (%s)\n", __func__, BUILD_NUMBER, BUILD_COMMIT);

    if (params.seed < 0) {
        params.seed = static_cast<int32_t>(time(NULL));
    }

    // fprintf(stderr, "%s: seed  = %d\n", __func__, params.seed);

    std::mt19937 rng(params.seed);
    if (params.random_prompt) {
        params.prompt = gpt_random_prompt(rng);
    }

    falcon_init_backend();

    falcon_context * ctx;
    g_ctx = &ctx;

    // load the model and apply lora adapter, if any
    ctx = falcon_init_from_gpt_params(params);
    if (ctx == NULL) {
        fprintf(stderr, "%s: error: unable to load model\n", __func__);
        return 1;
    }


    #if defined(GGML_USE_CUBLAS)
    // wait for cublas and show device information
    {
        ggml_cuda_print_gpu_status(ggml_cuda_get_system_gpu_status(),true);
    }
    #endif

    // print system information
    {
        fprintf(stderr, "\n");
        fprintf(stderr, "%s\n",
                 falcon_print_system_info(params.n_threads, std::thread::hardware_concurrency()));
    }

    // determine the maximum memory usage needed to do inference for the given n_batch and n_predict parameters
    // uncomment the "used_mem" line in llama.cpp to see the results
    if (params.mem_test) {
        falcon_prepare_buffers(ctx, params.n_batch, params.n_ctx);
        {
            const std::vector<falcon_token> tmp((int)params.n_batch, falcon_token_bos());
            falcon_eval(ctx, tmp.data(), (int)tmp.size(), 0, params.n_threads,params.debug_timings);
        }

        {
            const std::vector<falcon_token> tmp = { 0, };
            falcon_eval(ctx, tmp.data(), (int)tmp.size(), params.n_predict - 1, params.n_threads,params.debug_timings);
        }

        falcon_print_timings(ctx);
        llama_free(ctx);

        return 0;
    }

    // export the cgraph and exit
    if (params.export_cgraph) {
        falcon_eval_export(ctx, "ggllm.cpp");
        llama_free(ctx);

        return 0;
    }

    std::string path_session = params.path_prompt_cache;
    std::vector<falcon_token> session_tokens;

    if (!path_session.empty()) {
        fprintf(stderr, "%s: attempting to load saved session from '%s'\n", __func__, path_session.c_str());

        // fopen to check for existing session
        FILE * fp = std::fopen(path_session.c_str(), "rb");
        if (fp != NULL) {
            std::fclose(fp);

            session_tokens.resize(params.n_ctx);
            size_t n_token_count_out = 0;
            if (!llama_load_session_file(ctx, path_session.c_str(), session_tokens.data(), session_tokens.capacity(), &n_token_count_out)) {
                fprintf(stderr, "%s: error: failed to load session file '%s'\n", __func__, path_session.c_str());
                return 1;
            }
            session_tokens.resize(n_token_count_out);
            llama_set_rng_seed(ctx, params.seed);

            fprintf(stderr, "%s: loaded a session with prompt size of %d tokens\n", __func__, (int) session_tokens.size());
        } else {
            fprintf(stderr, "%s: session file does not exist, will create\n", __func__);
        }
    }

    // tokenize the prompt
    std::vector<falcon_token> embd_inp;

    if (params.interactive_first || params.instruct || !params.prompt.empty() || session_tokens.empty()) 
    {
        // Falcon does not have a dedicated bos token (bos==eos), so don't inject it here
        // auto start = ggml_time_us();
        embd_inp = ::falcon_tokenize(ctx, params.prompt, false);
        // auto end = ggml_time_us();
        // fprintf(stderr, "%s: tokenization took %0.3f ms\n", __func__, (end - start) / 1000.0);
        // fprintf(stderr, "%s: tokens processed: %d\n", __func__, (int) embd_inp.size());
        // fprintf(stderr, "%s: tokens/second : %0.3f\n", __func__, (embd_inp.size() / ((end - start) / 1000000.0)));
    } else {
        embd_inp = session_tokens;
    }

    const int n_ctx = falcon_n_ctx(ctx);

    if ((int) embd_inp.size() > n_ctx - 4) {
        fprintf(stderr, "%s: error: prompt is too long (%d tokens, max %d)\n", __func__, (int) embd_inp.size(), n_ctx - 4);
        return 1;
    }
    falcon_prepare_buffers(ctx, params.n_batch, embd_inp.size()+1);
    // debug message about similarity of saved session, if applicable
    size_t n_matching_session_tokens = 0;
    if (session_tokens.size()) {
        for (falcon_token id : session_tokens) {
            if (n_matching_session_tokens >= embd_inp.size() || id != embd_inp[n_matching_session_tokens]) {
                break;
            }
            n_matching_session_tokens++;
        }
        if (params.prompt.empty() && n_matching_session_tokens == embd_inp.size()) {
            fprintf(stderr, "%s: using full prompt from session file\n", __func__);
        } else if (n_matching_session_tokens >= embd_inp.size()) {
            fprintf(stderr, "%s: session file has exact match for prompt!\n", __func__);
        } else if (n_matching_session_tokens < (embd_inp.size() / 2)) {
            fprintf(stderr, "%s: warning: session file has low similarity to prompt (%zu / %zu tokens); will mostly be reevaluated\n",
                __func__, n_matching_session_tokens, embd_inp.size());
        } else {
            fprintf(stderr, "%s: session file matches %zu / %zu tokens of prompt\n",
                __func__, n_matching_session_tokens, embd_inp.size());
        }
    }

    // if we will use the cache for the full prompt without reaching the end of the cache, force
    // reevaluation of the last token token to recalculate the cached logits
    if (!embd_inp.empty() && n_matching_session_tokens == embd_inp.size() &&
            session_tokens.size() > embd_inp.size()) {
        session_tokens.resize(embd_inp.size() - 1);
    }

    // number of tokens to keep when resetting context
    if (params.n_keep < 0 || params.n_keep > (int) embd_inp.size() || params.instruct) {
        params.n_keep = (int)embd_inp.size();
    }

    // prefix & suffix for instruct mode and finetune modes
    std::vector<falcon_token> inp_system = {}; // system prompt
    std::vector<falcon_token> inp_pfx = {}; // prefix to user prompt
    std::vector<falcon_token> inp_sfx = {}; // suffix to user prompt
    std::vector<std::vector<falcon_token>> stopwords = {};

    if (params.stopwords.size())
    {
        std::string sw_token_str;
        std::vector<std::string> inp_system;
        std::stringstream stopwordStream(params.stopwords);
        std::vector<std::string> sw_token_list;
        while(std::getline(stopwordStream, sw_token_str, ',')) {
            sw_token_list.push_back(sw_token_str);
        }

        for (auto& sw_token : sw_token_list) {
            auto stopword_seq = ::falcon_tokenize(ctx, sw_token, false);
            stopwords.push_back(stopword_seq);
        }
    }
    #if 0
    {
        for (auto it = stopwords.begin(); it != stopwords.end(); ++it)
        {
            fprintf(stderr, "stopword: ");
            for (auto it2 = it->begin(); it2 != it->end(); ++it2)
            {
                const char *c_tk = falcon_token_to_str(ctx, *it2);
                if (*c_tk == '\n') c_tk="\\n";
                if (*c_tk == '\r') c_tk="\\r";
                fprintf(stderr, "%6d -> '%s', ", *it2, c_tk);
            }
            fprintf(stderr, "\n");
        }
    }
    #endif
        
    // auto detect finetune type if not specified - it's not that easy to do for most tunes
    // --alias can be used to force a fine tune, otherwise often just the filename is helpful
    if (params.finetune_type == FINETUNE_UNSPECIFIED)
    {
        params.finetune_type = falcon_detect_finetune(ctx,params.model);
    }
    if (params.instruct || params.enclose_finetune)
    {
        switch (params.finetune_type)
        {
            //FINETUNE_UNSPECIFIED, FINETUNE_NONE, FINETUNE_ALPACA, FINETUNE_OPENASSISTANT, FINETUNE_WIZARD, FINETUNE_FALCONINSTRUCT } t_finetune_type;
            case FINETUNE_ALPACA:
                inp_pfx = ::falcon_tokenize(ctx, "\n\n### Instruction:\n\n", false);
                inp_sfx = ::falcon_tokenize(ctx, "\n\n### Response:\n\n", false);
                if (params.system_prompt.size())
                inp_system = ::falcon_tokenize(ctx, params.system_prompt+"\n\n", false);
                break;
            case FINETUNE_OPENASSISTANT:
                //<|prefix_begin|>You are a helpful Assistant called Falcon<|prefix_end|>
                inp_pfx = ::falcon_tokenize(ctx, "<|prompter|>", false);
                inp_sfx = ::falcon_tokenize(ctx, "<|endoftext|><|assistant|>", false);
                if (params.system_prompt.size())
                {
                    inp_system = ::falcon_tokenize(ctx, "<|system|>Behavior:"+params.system_prompt+"<|endoftext|>", false);
                    // inp_system = ::falcon_tokenize(ctx, "<|prefix_begin|>"+params.system_prompt+"<|prefix_end|>", false);
                }
                break;
            case FINETUNE_WIZARD:
                inp_pfx = {};
                inp_sfx = ::falcon_tokenize(ctx, "\n### Response:", false);
                if (params.system_prompt.size())
                inp_system = ::falcon_tokenize(ctx, "### Behavior:\""+params.system_prompt+"\"\n", false);
                break;
            case FINETUNE_FALCONINSTRUCT:
                inp_pfx = ::falcon_tokenize(ctx, "User: \"", false);
                inp_sfx = ::falcon_tokenize(ctx, "\"\nAssistant:", false); // must not include space
                if (params.system_prompt.size())
                inp_system = ::falcon_tokenize(ctx, "System command: \""+params.system_prompt+"\"\n", false);
                break;
            default:
                inp_pfx = ::falcon_tokenize(ctx, ">>QUESTION<<", false);
                inp_sfx = ::falcon_tokenize(ctx, "\n>>ANSWER<<", false);
                if (params.system_prompt.size())
                inp_system = ::falcon_tokenize(ctx, ">>PREFIX<<"+params.system_prompt+"\n\n", false);
                if (params.stopwords.size() == 0)
                {
                    stopwords.push_back(::falcon_tokenize(ctx, ">>COMMENT<<", false));
                }
                break;
        }
    }
    // in instruct mode, we inject a prefix and a suffix to each input by the user
    if (params.instruct) {
        
        params.interactive_first = true;
        //params.antiprompt.push_back("### Instruction:\n\n");
    }

    // enable interactive mode if interactive start is specified
    if (params.interactive_first) {
        params.interactive = true;
    }

    // determine newline token
    //auto llama_token_newline = ::falcon_tokenize(ctx, "\n", false);
    auto falcon_token_newline = std::vector<falcon_token>(193);

    if (params.verbose_prompt) {
        fprintf(stderr, "\n");
        fprintf(stderr, "%s: prompt: '%s'\n", __func__, params.prompt.c_str());
        fprintf(stderr, "%s: number of tokens in prompt = %zu\n", __func__, embd_inp.size());
        for (int i = 0; i < (int) embd_inp.size(); i++) {
            const char *c_tk = falcon_token_to_str(ctx, embd_inp[i]);
            if (*c_tk == '\n') c_tk="\\n";
            if (*c_tk == '\r') c_tk="\\r";
            fprintf(stderr, "%6d -> '%s'\n", embd_inp[i], c_tk);
        }
        if (params.n_keep > 0) {
        fprintf(stderr, "%s: static prompt based on n_keep: '", __func__);
            for (int i = 0; i < params.n_keep; i++) {
                            const char *c_tk = falcon_token_to_str(ctx, embd_inp[i]);
            if (*c_tk == '\n') c_tk="\\n";
            if (*c_tk == '\r') c_tk="\\r";
                fprintf(stderr, "%s", c_tk);
            }
            fprintf(stderr, "'\n");
        }
        fprintf(stderr, "\n");
    }

    if (params.interactive) {
#if defined (__unix__) || (defined (__APPLE__) && defined (__MACH__))
        struct sigaction sigint_action;
        sigint_action.sa_handler = sigint_handler;
        sigemptyset (&sigint_action.sa_mask);
        sigint_action.sa_flags = 0;
        sigaction(SIGINT, &sigint_action, NULL);
#elif defined (_WIN32)
        auto console_ctrl_handler = +[](DWORD ctrl_type) -> BOOL {
            return (ctrl_type == CTRL_C_EVENT) ? (sigint_handler(SIGINT), true) : false;
        };
        SetConsoleCtrlHandler(static_cast<PHANDLER_ROUTINE>(console_ctrl_handler), true);
#endif

        fprintf(stderr, "%s: interactive mode on.\n", __func__);

        if (params.antiprompt.size()) {
            for (auto antiprompt : params.antiprompt) {
                fprintf(stderr, "Reverse prompt: '%s'\n", antiprompt.c_str());
            }
        }

        if (!params.input_prefix.empty()) {
            fprintf(stderr, "Input prefix: '%s'\n", params.input_prefix.c_str());
        }

        if (!params.input_suffix.empty()) {
            fprintf(stderr, "Input suffix: '%s'\n", params.input_suffix.c_str());
        }
    }

size_t prompt_size = embd_inp.size();
if (params.enclose_finetune || params.instruct)
{
    prompt_size+=inp_pfx.size()+inp_sfx.size()+inp_system.size();
}

fprintf(stderr, "+------------+-------+-------+-------+-------+-------+-------+-------+-------+------+------+--------+---------+\n");
fprintf(stderr, "| %10s | %5s | %4s | %4s | %4s | %4s | %4s | %4s | %4s | %4s | %4s | %4s | %4s |\n", 
                "Sampling","rpt_n","rpt_p","prs_p","frq_p","top_k","tfs_z", "top_p", "typ_p", "temp", "miro", "mir_lr", "mir_ent");
fprintf(stderr, "+------------+-------+-------+-------+-------+-------+-------+-------+-------+------+------+--------+---------+\n");
fprintf(stderr, "|            | %5d | %.3f | %.3f | %.3f | %5d | %.3f | %.3f | %.3f | %.2f | %4d | %.4f | %.5f |\n", 
                params.repeat_last_n, params.repeat_penalty, params.presence_penalty, params.frequency_penalty, params.top_k, params.tfs_z, params.top_p, params.typical_p, params.temp, params.mirostat, params.mirostat_eta, params.mirostat_tau);
fprintf(stderr, "+============+=======+=======+=======+=======+=======+=======+-------+-------+------+------+--------+---------+\n");

fprintf(stderr, "| %10s | %5s | %5s | %5s | %5s | %13s | %20s | %4s |\n", 
                "Generation", "Ctx", "Batch", "Keep","Prom.","Seed","Finetune", "Stop");
fprintf(stderr, "+------------+-------+-------+-------+-------+---------------+----------------------+------+\n");  
fprintf(stderr, "|            | %5d | %5d | %5d | %5zu | %13d | %20s | #%3lld |\n",
                n_ctx, params.n_batch, params.n_keep, prompt_size,params.seed,FINETUNE_NAME[params.finetune_type], ((params.logit_bias[falcon_token_eos()] == -INFINITY)?0:1)+stopwords.size());
fprintf(stderr, "+------------+-------+-------+-------+-------+---------------+----------------------+------+\n");  

    if (n_ctx < (int)(params.n_predict + embd_inp.size())) {
        fprintf(stderr, "%s: Warning: context is smaller than expected generation, will cause delays\n", __func__);
    }

    fprintf(stderr, "\n\n");

    // TODO: replace with ring-buffer
    std::vector<falcon_token> last_n_tokens(n_ctx);
    std::fill(last_n_tokens.begin(), last_n_tokens.end(), 0);

    if (params.interactive) {
        const char *control_message;
        if (con_st.multiline_input) {
            control_message = " - To return control to ggLLM, end your input with '\\'.\n"
                              " - To return control without starting a new line, end your input with '/'.\n";
        } else {
            control_message = " - Press Return to return control to ggLLM.\n"
                              " - To return control without starting a new line, end your input with '/'.\n"
                              " - If you want to submit another line, end your input with '\\'.\n";
        }
        fprintf(stderr, "== Running in interactive mode. ==\n"
#if defined (__unix__) || (defined (__APPLE__) && defined (__MACH__)) || defined (_WIN32)
               " - Press Ctrl+C to interject at any time.\n"
#endif
               "%s\n", control_message);

        is_interacting = params.interactive_first;
    }

    bool is_antiprompt        = false;
    bool input_echo           = true;
    bool need_to_save_session = !path_session.empty() && n_matching_session_tokens < embd_inp.size();

    int n_past             = 0;
    int n_remain           = params.n_predict;
    int n_consumed         = 0;
    int n_session_consumed = 0;

    // the first thing we will do is to output the prompt, so set color accordingly
    console_set_color(con_st, CONSOLE_COLOR_PROMPT);

    std::vector<falcon_token> embd;

    

    // do one empty run to warm up the model
    {
        const std::vector<falcon_token> tmp = { falcon_token_bos(), };
        falcon_eval(ctx, tmp.data(), tmp.size(), 0, params.n_threads,0);
        llama_reset_timings(ctx);
    }


    while ((n_remain != 0 && !is_antiprompt) || params.interactive) 
    {
        // predict
        if (embd.size() > 0) 
        {
            // Note: n_ctx - 4 here is to match the logic for commandline prompt handling via
            // --prompt or --file which uses the same value.
            auto max_embd_size = n_ctx - 4;
            // Ensure the input doesn't exceed the context size by truncating embd if necessary.
            if ((int)embd.size() > max_embd_size) {
                auto skipped_tokens = embd.size() - max_embd_size;
                console_set_color(con_st, CONSOLE_COLOR_ERROR);
                printf("<<input too long: skipped %zu token%s>>", skipped_tokens, skipped_tokens != 1 ? "s" : "");
                console_set_color(con_st, CONSOLE_COLOR_DEFAULT);
                fflush(stdout);
                embd.resize(max_embd_size);
            }

            // infinite text generation via context swapping
            // if we run out of context:
            // - take the n_keep first tokens from the original prompt (via n_past)
            // - take half of the last (n_ctx - n_keep) tokens and recompute the logits in batches
            if (n_past + (int) embd.size() > n_ctx) {
                const int n_left = n_past - params.n_keep;

                n_past = params.n_keep;

                // insert n_left/2 tokens at the start of embd from last_n_tokens
                embd.insert(embd.begin(), last_n_tokens.begin() + n_ctx - n_left/2 - embd.size(), last_n_tokens.end() - embd.size());
                // stop saving session if we run out of context
                path_session.clear();

                //printf("\n---\n");
                //printf("resetting: '");
                //for (int i = 0; i < (int) embd.size(); i++) {
                //    printf("%s", falcon_token_to_str(ctx, embd[i]));
                //}
                //printf("'\n");
                //printf("\n---\n");
            }

            // try to reuse a matching prefix from the loaded session instead of re-eval (via n_past)
            if (n_session_consumed < (int) session_tokens.size()) {
                size_t i = 0;
                for ( ; i < embd.size(); i++) {
                    if (embd[i] != session_tokens[n_session_consumed]) {
                        session_tokens.resize(n_session_consumed);
                        break;
                    }

                    n_past++;
                    n_session_consumed++;

                    if (n_session_consumed >= (int) session_tokens.size()) {
                        ++i;
                        break;
                    }
                }
                if (i > 0) {
                    embd.erase(embd.begin(), embd.begin() + i);
                }
            }
            // evaluate tokens in batches
            // embd is typically prepared beforehand to fit within a batch, but not always
            for (int i = 0; i < (int) embd.size(); i += params.n_batch) 
            {
                int n_eval = (int) embd.size() - i;
                if (n_eval > params.n_batch) {
                    n_eval = params.n_batch;
                }
                int debug_timings = params.debug_timings;
                if (n_remain == 1 && debug_timings == 2) debug_timings = 3; // we have no access to the last information in eval()
                if (falcon_eval(ctx, &embd[i], n_eval, n_past, params.n_threads,debug_timings)) {
                    fprintf(stderr, "%s : failed to eval\n", __func__);
                    return 1;
                }
                n_past += n_eval;
            }
            if (embd.size() > 0 && !path_session.empty()) {
                session_tokens.insert(session_tokens.end(), embd.begin(), embd.end());
                n_session_consumed = session_tokens.size();
            }
        } // if (embd.size() > 0)

        embd.clear();

        if ((int) embd_inp.size() <= n_consumed && !is_interacting)  // sample for next generation
        {
            // out of user input, sample next token
            const float   temp            = params.temp;
            const int32_t top_k           = params.top_k <= 0 ? falcon_n_vocab(ctx) : params.top_k;
            const float   top_p           = params.top_p;
            const float   tfs_z           = params.tfs_z;
            const float   typical_p       = params.typical_p;
            const int32_t repeat_last_n   = params.repeat_last_n < 0 ? n_ctx : params.repeat_last_n;
            const float   repeat_penalty  = params.repeat_penalty;
            const float   alpha_presence  = params.presence_penalty;
            const float   alpha_frequency = params.frequency_penalty;
            const int     mirostat        = params.mirostat;
            const float   mirostat_tau    = params.mirostat_tau;
            const float   mirostat_eta    = params.mirostat_eta;
            const bool    penalize_nl     = params.penalize_nl;

            // optionally save the session on first sample (for faster prompt loading next time)
            if (!path_session.empty() && need_to_save_session && !params.prompt_cache_ro) {
                need_to_save_session = false;
                llama_save_session_file(ctx, path_session.c_str(), session_tokens.data(), session_tokens.size());
            }

            falcon_token id = 0;

            {
                auto logits  = falcon_get_logits(ctx);
                auto n_vocab = falcon_n_vocab(ctx);

                // Apply params.logit_bias map
                for (auto it = params.logit_bias.begin(); it != params.logit_bias.end(); it++) {
                    logits[it->first] += it->second;
                }

                std::vector<falcon_token_data> candidates;
                candidates.reserve(n_vocab);
                for (falcon_token token_id = 0; token_id < n_vocab; token_id++) {
                    candidates.emplace_back(falcon_token_data{token_id, logits[token_id], 0.0f});
                }

                falcon_token_data_array candidates_p = { candidates.data(), candidates.size(), false };

                // Apply penalties
                float nl_logit = logits[falcon_token_nl()];
                auto last_n_repeat = std::min(std::min((int)last_n_tokens.size(), repeat_last_n), n_ctx);
                llama_sample_repetition_penalty(ctx, &candidates_p,
                    last_n_tokens.data() + last_n_tokens.size() - last_n_repeat,
                    last_n_repeat, repeat_penalty);
                llama_sample_frequency_and_presence_penalties(ctx, &candidates_p,
                    last_n_tokens.data() + last_n_tokens.size() - last_n_repeat,
                    last_n_repeat, alpha_frequency, alpha_presence);
                if (!penalize_nl) {
                    logits[falcon_token_nl()] = nl_logit;
                }

                if (temp <= 0) {
                    // Greedy sampling
                    id = llama_sample_token_greedy(ctx, &candidates_p);
                } else {
                    if (mirostat == 1) {
                        static float mirostat_mu = 2.0f * mirostat_tau;
                        const int mirostat_m = 100;
                        llama_sample_temperature(ctx, &candidates_p, temp);
                        id = llama_sample_token_mirostat(ctx, &candidates_p, mirostat_tau, mirostat_eta, mirostat_m, &mirostat_mu);
                    } else if (mirostat == 2) {
                        static float mirostat_mu = 2.0f * mirostat_tau;
                        llama_sample_temperature(ctx, &candidates_p, temp);
                        id = llama_sample_token_mirostat_v2(ctx, &candidates_p, mirostat_tau, mirostat_eta, &mirostat_mu);
                    } else {
                        // Temperature sampling
                        llama_sample_top_k(ctx, &candidates_p, top_k, 1);
                        llama_sample_tail_free(ctx, &candidates_p, tfs_z, 1);
                        llama_sample_typical(ctx, &candidates_p, typical_p, 1);
                        llama_sample_top_p(ctx, &candidates_p, top_p, 1);
                        llama_sample_temperature(ctx, &candidates_p, temp);
                        id = llama_sample_token(ctx, &candidates_p);
                    }
                }
                // printf("`%d`", candidates_p.size);

                last_n_tokens.erase(last_n_tokens.begin());
                last_n_tokens.push_back(id);
            }

            // replace end of text token with newline token when in interactive mode // todo: openassistant and wizard handling
            #if 0
            // disabled for now - some finetunes actually need that token - audit if that is needed by normal use
            if (id == falcon_token_eos() && params.interactive && !params.instruct) {
                id = falcon_token_newline.front();
                if (params.antiprompt.size() != 0) {
                    // tokenize and inject first reverse prompt
                    const auto first_antiprompt = ::falcon_tokenize(ctx, params.antiprompt.front(), false);
                    embd_inp.insert(embd_inp.end(), first_antiprompt.begin(), first_antiprompt.end());
                }
            }
            #endif

            // add it to the context
            embd.push_back(id);

            // echo this to console
            input_echo = true;

            // decrement remaining sampling budget
            --n_remain;
        } else 
        {
            // some user input remains from prompt or interaction, forward it to processing
            if (n_past == 0)
            {
                if ( !params.interactive && params.enclose_finetune && (inp_pfx.size() || inp_sfx.size()) )
                {
                    // enclose finetune - non interactive mode
                    if (inp_pfx.size())
                    {
                        embd_inp.insert(embd_inp.begin(), inp_pfx.begin(), inp_pfx.end());
                    }
                    if (inp_system.size())
                    {
                        embd_inp.insert(embd_inp.begin(), inp_system.begin(), inp_system.end());
                    }
                    if (inp_sfx.size())
                    {
                        embd_inp.insert(embd_inp.end(), inp_sfx.begin(), inp_sfx.end());
                    }
                }
            }
            while ((int) embd_inp.size() > n_consumed) {
                embd.push_back(embd_inp[n_consumed]);
                last_n_tokens.erase(last_n_tokens.begin());
                last_n_tokens.push_back(embd_inp[n_consumed]);
                ++n_consumed;
                if ((int) embd.size() >= params.n_batch) {
                    break;
                }
            }
        }

        bool stopword_fulfilled = false;
        // stopwords
        if (!embd.empty()) 
        {
            
            for (const auto& stopword : stopwords) {
                if (embd.size() < stopword.size()) {
                    continue; // if embd is smaller than stopword, skip this iteration
                }
                stopword_fulfilled = true; // initially assume stopword is fulfilled
                for (size_t i = 0; i < stopword.size(); ++i) {
                    if (embd[embd.size() - i - 1] != stopword[stopword.size() - i - 1]) {
                        stopword_fulfilled = false;
                        break;
                    }
                }
                if (stopword_fulfilled) {
                    break;
                }
            }
            if (stopword_fulfilled) 
            {
                if (params.verbose_prompt) 
                    fprintf(stderr, " [stopword]\n");
                break;
            }
        }
        // display text
        if (input_echo) 
        {
            for (auto id : embd) {
                if (params.instruct && id == falcon_token_eos()) {
                    id = falcon_token_nl();
                }
                printf("%s", falcon_token_to_str(ctx, id));
            }
            fflush(stdout);
        }
        // reset color to default if we there is no pending user input
        if (input_echo && (int)embd_inp.size() == n_consumed) 
        {
            console_set_color(con_st, CONSOLE_COLOR_DEFAULT);
        }

        // if not currently processing queued inputs;
        if ((int) embd_inp.size() <= n_consumed) 
        {

            // check for reverse prompt
            if (params.antiprompt.size()) 
            {
                std::string last_output;
                for (auto id : last_n_tokens) {
                    last_output += falcon_token_to_str(ctx, id);
                }

                is_antiprompt = false;
                // Check if each of the reverse prompts appears at the end of the output.
                // If we're not running interactively, the reverse prompt might be tokenized with some following characters
                // so we'll compensate for that by widening the search window a bit.
                for (std::string & antiprompt : params.antiprompt) 
                {
                    size_t extra_padding = params.interactive ? 0 : 2;
                    size_t search_start_pos = last_output.length() > static_cast<size_t>(antiprompt.length() + extra_padding)
                        ? last_output.length() - static_cast<size_t>(antiprompt.length() + extra_padding)
                        : 0;

                    if (last_output.find(antiprompt.c_str(), search_start_pos) != std::string::npos) {
                        if (params.interactive) {
                            is_interacting = true;
                            console_set_color(con_st, CONSOLE_COLOR_USER_INPUT);
                        }
                        is_antiprompt = true;
                        fflush(stdout);
                        break;
                    }
                }
            }
            std::string buffer;
            // get user interactive input
            if (n_past >= 0 && is_interacting) 
            {
                if (params.instruct) {
                    printf("\n> ");
                }

                
                if (!params.input_prefix.empty()) {
                    buffer += params.input_prefix;
                    printf("%s", buffer.c_str());
                }

                std::string line;
                bool another_line = true;
                do {
                    another_line = console_readline(con_st, line);
                    buffer += line;
                } while (another_line);

                // done taking input, reset color
                console_set_color(con_st, CONSOLE_COLOR_DEFAULT);
            }
            if (n_past >= 0 && (is_interacting)) 
            {
                // Add tokens to embd only if the input buffer is non-empty
                // Entering a empty line lets the user pass control back
                if (buffer.length() > 1) {
                    // append input suffix if any
                    if (!params.input_suffix.empty()) {
                        buffer += params.input_suffix;
                        printf("%s", params.input_suffix.c_str());
                    }

                    // instruct mode: insert instruction prefix
                    if (params.instruct && !is_antiprompt) {
                        n_consumed = embd_inp.size();
                        embd_inp.insert(embd_inp.end(), inp_pfx.begin(), inp_pfx.end());
                    }

                    auto line_inp = ::falcon_tokenize(ctx, buffer, false);
                    embd_inp.insert(embd_inp.end(), line_inp.begin(), line_inp.end());

                    // instruct mode: insert response suffix
                    if (params.instruct) {
                        embd_inp.insert(embd_inp.end(), inp_sfx.begin(), inp_sfx.end());
                    }

                    n_remain -= line_inp.size(); // ugh - don't like ignoring the prompts. needs a audit
                }

                // system prompt injection
                if (params.interactive && params.antiprompt.size()==0 && n_consumed == 0 && inp_system.size()) 
                {
                    embd_inp.insert(embd_inp.begin(), inp_system.begin(), inp_system.end());
                } 
                if (embd_inp.size())
                {
                    if (params.verbose_prompt) 
                    {
                        fprintf(stderr, "\n");
                        fprintf(stderr, "%s: number of tokens in prompt = %zu\n", __func__, embd_inp.size());
                        for (int i = 0; i < (int) embd_inp.size(); i++) {
                            const char *c_tk = falcon_token_to_str(ctx, embd_inp[i]);
                            if (*c_tk == '\n') c_tk="\\n";
                            if (*c_tk == '\r') c_tk="\\r";
                            fprintf(stderr, "%6d -> '%s'\n", embd_inp[i], c_tk);
                        }
                    }
                }

                input_echo = false; // do not echo this again
            } 

            if (n_past >= 0) {
                is_interacting = false;
            }
        }
        #if 0
        // debug dump entire embd now:
        if (params.verbose_prompt) {
            fprintf(stderr, "\n\n%s: number of tokens in embd = %zu\n", __func__, embd.size());
            for (int i = 0; i < (int) embd.size(); i++) {
                const char *c_tk = falcon_token_to_str(ctx, embd[i]);
                if (*c_tk == '\n') c_tk="\\n";
                if (*c_tk == '\r') c_tk="\\r";
                fprintf(stderr, "%6d -> '%s'\n", embd[i], c_tk);
            }
        }
        #endif

        // end of text token
        if (!embd.empty() && embd.back() == falcon_token_eos()) {
            if (params.instruct) {
                is_interacting = true;
            } else {
                if (params.verbose_prompt)
                    fprintf(stderr, " [end of text]\n");
                // if we are in the prompt ingestion we will not stop
                if (n_past > (int)embd_inp.size()) {
                    break;
                }
            }
        }

        

        // In interactive mode, respect the maximum number of tokens and drop back to user input when reached.
        if (params.interactive && n_remain <= 0 && params.n_predict != -1) {
            n_remain = params.n_predict;
            is_interacting = true;
        }
    }

    if (!path_session.empty() && params.prompt_cache_all && !params.prompt_cache_ro) {
        fprintf(stderr, "\n%s: saving final output to session file '%s'\n", __func__, path_session.c_str());
        llama_save_session_file(ctx, path_session.c_str(), session_tokens.data(), session_tokens.size());
    }

    falcon_print_timings(ctx);
    llama_free(ctx);

    return 0;
}
