#ifndef __GGML_COSYVOICE_H__
#define __GGML_COSYVOICE_H__

#include "ggml_module.h"
#include <set>

#define COSYVOICE_MAX_NODES 8192

struct cosyvoice_params_t
{
    uint32_t n_batch;
    uint32_t n_max_seq;
    uint32_t seed;

    bool llm_use_flash_attn;
    bool flow_use_flash_attn;
};


struct cosyvoice_model_t
{
#if 0
    ggml_context *ctx;
    ggml_backend_buffer_t buf_weights;

    //prompt

    CausalMaskedDiffWithDiT flow;
    CausalHiFTGenerator hift;
    CosyVoice3LM llm;

    ggml_type k_type;
    ggml_type v_type;

    std::set<int> stop_tokens;
#endif
};

struct cosyvoice_state_t
{
    ggml_backend_sched_t sched = nullptr;

    ggml_context *ctx_cache;
    ggml_backend_buffer_t buf_cache;

    ggml_context *ctx_build;
    void *build_buf;
    int build_buf_size;
};


int load_cosyvoice_model(struct ggml_handle_t *ggml_handle, const char *model_data, int model_size);
int cosyvoice_inference(struct ggml_handle_t *ggml_handle, matrix_t **input_matrix,
                        matrix_t **output_matrix);

void unload_cosyvoice_model(struct ggml_handle_t *ggml_handle);


#endif //  __GGML_COSYVOICE_H__
