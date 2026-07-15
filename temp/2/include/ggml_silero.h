#ifndef __GGML_SILERO_H__
#define __GGML_SILERO_H__

#include "ggml_module.h"

#define SILERO_MAX_NODES 8192
#define VAD_CHUNK_SIZE 640

struct silero_model_t
{
    ggml_context *ctx;
    ggml_backend_buffer_t buf_weights;

    STFT stft;
    std::vector<Conv1d> encoders;
    LSTM rnn;
    Conv1d decoder;
};

struct silero_state_t
{
    ggml_backend_sched_t sched = nullptr;

    ggml_context *ctx_cache;
    ggml_backend_buffer_t buf_cache;
    struct ggml_tensor * vad_lstm_hidden_state;
    struct ggml_tensor * vad_lstm_context;

    ggml_context *ctx_build;
    void *build_buf;
    int build_buf_size;

    float chunk_cache[64];
    int frame_count;
};

struct silero_params_t
{
    uint32_t n_encoder_layer;
    uint32_t lstm_state_dim;

    uint32_t n_batch;

    float threshold;
    float neg_threshold;
    int32_t min_speech_duration_ms;
    int32_t max_speech_duration_ms;
    int32_t min_silence_duration_ms;
    int32_t speech_pad_ms;
};

/* vad_silero */
int load_silero_model(struct ggml_handle_t *ggml_handle, const char *model_data, int model_size);
int silero_inference(struct ggml_handle_t *ggml_handle, matrix_t **input_matrix,
                        matrix_t **output_matrix);

void unload_silero_model(struct ggml_handle_t *ggml_handle);


#endif //  __GGML_SILERO_H__
