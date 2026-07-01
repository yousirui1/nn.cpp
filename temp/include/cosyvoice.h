#ifndef __COSYVOICE_H__
#define __COSYVOICE_H__

#include <set>

#include "ggml.h"
#include "ggml-cpu.h"
#include "nn.h"
#include "ggml_module.h"
#include "ggml-cpp.h"
#include "llm_kv_cache.h"

typedef struct cosyvoice_context_params
{
    bool llm_use_flash_attn;  ///< Use flash attention if supported by the backend. Falls back to regular attention otherwise.
    bool flow_use_flash_attn; ///< Use flash attention for the Flow module if supported.

    enum ggml_type llm_kv_cache_type;           ///< The data type of the KV cache in the LLM module.
    bool llm_allow_kv_cache_fallback; ///< If true, fall back to a Flash Attention-compatible KV cache type when the requested one is unsupported.
    nn_inference_buffer_policy_e inference_buffer_policy;     ///< Controls how inference buffers are allocated and reused, which affects performance and memory usage.

    uint32_t n_batch;   ///< Batch size used by inference kernels.
    uint32_t n_max_seq; ///< Maximum supported sequence length.
    uint32_t seed;      ///< Seed used by the built-in sampler and noise generator RNG.
    //cosyvoice_builtin_sampler_rng_policy_t builtin_sampler_rng_policy; ///< Controls how the built-in sampler RNG evolves across LLM sessions. Ignored when `sampler` is not null.

    // Sampling overrides
    //cosyvoice_sampler_t sampler;     ///< Optional custom sampler. Pass null to use the built-in sampler.    void*               sampler_ctx; ///<User context for `sampler`. Ignored when using the built-in sampler.    

}cosyvoice_context_params_t;


struct cosyvoice_model
{
    cosyvoice_model(ggml_backend_t backend, const cosyvoice_context_params_t& params);

    CausalMaskedDiffWithDiT flow;
    CausalHiFTGenerator hift;
    Qwen2LM llm;

    void load(gguf_loader& loader);

    bool llm_decode(ggml_type type, const void* data);
    bool llm_prefill(ggml_type type, const void* data, uint32_t seq_len);

    bool llm_is_stop_token(int token_id);

    const ggml_tensor* get_word_token_embed_weight();
    const ggml_tensor* get_speech_token_embed_weight();

#if 0
    uint32_t get_hift_rand_ini_len();
    void set_hift_rand_ini(const float* data);

    bool llm_job(const int* text, uint32_t text_len, cosyvoice_prompt_t prompt);
#endif

    //bool token2wav(const int* token_ids, uint32_t n_tokens, float speed, cosyvoice_prompt_t prompt, cosyvoice_generated_speech_ptr result);

    //void get_memory_usage(cosyvoice_memory_usage_t* usage);

    void empty_buffer_cache();
    void reset_shared_buffer(ggml_backend_buffer* new_buffer);

    cosyvoice_context_params_t params;
    ggml_status status;

    ggml_context_ptr ctx;

    ggml_backend_ptr backend;
    ggml_backend_ptr cpu_backend;
    ggml_backend_buffer_ptr buffer;
    ggml_backend_sched_ptr sched;

    ggml_cgraph* gf;
    ggml_context_ptr ctx0;

    ggml_tensor* llm_input;
    ggml_tensor* llm_probs;
    ggml_tensor* position_ids;
    ggml_tensor* causal_mask;

    ggml_context_ptr ctx1;
    ggml_backend_buffer_ptr token2wav_buffer;

    llm_kv_cache *kv_cache;

    std::vector<int> tokens;
    std::unique_ptr<int> full_position_ids;
    std::unique_ptr<ggml_fp16_t[]> causal_mask_buffer;

    std::set<int> stop_tokens;

    uint32_t orig_max_seq_len;
    ggml_type kv_type;

    ggml_backend_op_capabilities op_caps;

    //std::unique_ptr<float[]> rand_noise;
    std::unique_ptr<char> batch_buffer;
    //std::unique_ptr<float[]> nucleus_probs;
    std::unique_ptr<float[]> probs;
    std::unique_ptr<char[]> instruction_prefix;

    ggml_backend_buffer_ptr kv_buffer;
};


struct cosyvoice_context : cosyvoice_model
{
    cosyvoice_context(const cosyvoice_context_params_t& params, ggml_backend_t backend) :
        cosyvoice_model(backend, params) {}

    // quant
    //bool llm_job();
};


typedef struct cosyvoice_context *cosyvoice_context_t;



void init_cosyvoice_backend();
void cosyvoice_default_params(cosyvoice_context_params_t *params);

cosyvoice_context_t load_cosyvoice_model(const char *filename,
                                const cosyvoice_context_params_t params,
                                ggml_backend_t backend,
                                uint32_t n_threads,
                                uint32_t reserved);




#endif //  __COSYVOICE_H__
