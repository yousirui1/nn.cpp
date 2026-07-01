#ifndef __SILERO_H__
#define __SILERO_H__

#include "ggml_module.h"
#include "ggml.h"
#include "ggml-cpp.h"



typedef struct silero_context_params
{
	uint32_t n_encoder_layer;
    uint32_t lstm_state_dim;

    uint32_t n_batch;   
    uint32_t n_max_seq; 

}silero_context_params_t;

struct silero_model
{
    silero_model(ggml_backend_t backend, const silero_context_params_t& params);

    STFT stft;
    std::vector<Conv1d> encoders;
    LSTM rnn;
    Conv1d decoder;

    void load(gguf_loader& loader);

    void build_graph();

    int inference();

    void empty_buffer_cache();

    silero_context_params_t params;
    ggml_status status;

    ggml_context_ptr ctx;

    ggml_backend_ptr backend;
    ggml_backend_ptr cpu_backend;
    ggml_backend_buffer_ptr buffer;
    ggml_backend_sched_ptr sched;

    ggml_cgraph* gf;

    ggml_tensor* lstm_context;
    ggml_tensor* lstm_hidden_state;

    ggml_tensor *input;

    ggml_type kv_type;

    ggml_backend_op_capabilities op_caps;

    std::unique_ptr<char> batch_buffer;
};


struct silero_context : silero_model
{
    silero_context(const silero_context_params_t& params, ggml_backend_t backend) :
        silero_model(backend, params) {}

    // quant
    //bool llm_job();
};


typedef struct silero_context *silero_context_t;



void init_silero_backend();
void silero_default_params(silero_context_params_t *params);

silero_context_t load_silero_model(const char *filename,
                                const silero_context_params_t params,
                                ggml_backend_t backend,
                                uint32_t n_threads,
                                uint32_t reserved);


#endif //  __SILERO_H__
