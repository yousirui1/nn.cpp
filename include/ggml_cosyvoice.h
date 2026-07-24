#ifndef __GGML_COSYVOICE_H__
#define __GGML_COSYVOICE_H__

#include <set>

struct sampling_param_t
{
    int top_k;
    float top_p;
    int win_size;
    float tau_r;
};


struct cosyvoice_params_t
{
    bool llm_use_flash_attn; 
    bool flow_use_flash_attn;
    bool llm_allow_kv_cache_fallback;

    uint32_t n_batch;  
    uint32_t n_llm_max_seq; 
    uint32_t seed;      

    int n_workers;

    //sampler 
    //sampler_ext
    //sampler_ctx

    // llm_kv_cache_type
};



struct cosyvoice_model_t
{
    ggml_context *ctx;
    ggml_backend_buffer_t buf_weights;

    CausalMaskedDiffWithDiT flow;
    CausalHiFTGenerator hift;
    CosyVoice3LM llm;

    ggml_type k_type;
    ggml_type v_type;
        

    sampling_param_t sampling;

    std::set<int> stop_tokens;
    std::unique_ptr<char []>instruction_prefix;
    std::unique_ptr<char []>architecture;
};

#endif //  __GGML_COSYVOICE_H__
