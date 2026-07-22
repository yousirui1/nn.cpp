#pragma once

#include "cosyvoice_modules.h"
#include "cosyvoice_lowlevel.h"
#include "cosyvoice_interface.h"
#include "cosyvoice_llm_kv_cache.h"

#include <shared_mutex>
#include <ggml-cpp.h>
#include <set>
#include <random>

// CosyVoice2 共享上下文结构
struct cosyvoice_model_2_shared
{
    // CosyVoice2 使用 CausalMaskedDiffWithXvec 而非 DiT
    CausalMaskedDiffWithXvec flow;
    CausalHiFTGenerator hift;
    
    // CosyVoice2 LLM 结构
    CosyVoice2LM llm;

    ggml_type k_type;
    ggml_type v_type;

    // CosyVoice2 的 stop tokens 范围更大 (speech_token_size + 200)
    std::set<int> stop_tokens;
    
    // Token 集大小
    uint32_t speech_token_size;
};

// CosyVoice2 worker 上下文扩展
struct cosyvoice_2_worker_context
{
    cosyvoice_2_worker_context();
    ~cosyvoice_2_worker_context() = default;

    ggml_context_ptr ctx1;
    ggml_backend_buffer_ptr token2wav_buffer;

    uint32_t orig_max_seq_len;
    
    // 流式推理专用缓冲区
    std::unique_ptr<float[]> flow_cond_buffer;
    uint32_t flow_cond_len;
};

// CosyVoice2 模型类
struct cosyvoice_model_2 : cosyvoice_model
{
    cosyvoice_model_2(ggml_backend_t backend, const cosyvoice_context_params_v2_cpp& params);
    ~cosyvoice_model_2();
    
    void load(gguf_loader& loader);

    bool set_worker_no(uint32_t worker_no);

    // LLM 操作
    bool llm_decode(ggml_type type, const void* data);
    void llm_prepare_probs(bool allow_stop_tokens);
    bool llm_prefill(ggml_type type, const void* data, uint32_t seq_len);

    bool llm_is_stop_token(int token_id);

    uint32_t get_sample_rate();

    const ggml_tensor* get_word_token_embed_weight();
    const ggml_tensor* get_speech_token_embed_weight();

    uint32_t get_hift_rand_ini_len();
    void set_hift_rand_ini(const float* data);

    bool llm_job(const int* text, uint32_t text_len, cosyvoice_prompt_t prompt);
    bool token2wav(const int* token_ids, uint32_t n_tokens, float speed, cosyvoice_prompt_t prompt, cosyvoice_generated_speech_ptr result);

    void empty_buffer_cache();
    void get_memory_usage(cosyvoice_memory_usage_t* usage);
    void get_total_memory_usage(cosyvoice_memory_usage_t* usage);
    void reset_shared_buffer(ggml_backend_buffer* new_buffer);

    // CosyVoice2 流式推理接口
    bool llm_prefill_chunk(ggml_type type, const void* data, uint32_t seq_len, bool is_final);
    bool llm_decode_step(ggml_type type, const void* data, int* sampled_token);
    bool flow_encode_chunk(const int* token_ids, uint32_t n_tokens, float* mel_spec, uint32_t* mel_len);
    void streaming_reset();

    cosyvoice_model_2_shared* cv2_shared;
    cosyvoice_2_worker_context* cv2_workers;
    cosyvoice_2_worker_context* cv2_worker;
    
    // 流式状态
    struct {
        uint32_t chunk_count;
        bool is_streaming;
        uint32_t total_tokens;
    } streaming_state;
};
