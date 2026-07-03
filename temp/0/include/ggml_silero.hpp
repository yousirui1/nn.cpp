#ifndef __GGML_SILERO_H__
#define __GGML_SILERO_H__

#include "ggml_module.hpp"

#define SILERO_MAX_NODES 8192
#define VAD_CHUNK_SIZE 640

struct silero_model_t
{
    STFT stft;
    std::vector<Conv1d> encoders;
    LSTM rnn;
    Conv1d decoder;
};

struct silero_state_t
{
    ggml_backend_sched_t sched = nullptr;
    std::vector<uint8_t> meta;

    struct ggml_tensor * vad_lstm_hidden_state;
    struct ggml_tensor * vad_lstm_context;

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


#endif //  __GGML_SILERO_H__
