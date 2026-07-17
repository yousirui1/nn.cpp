#ifndef __GGML_FSMN_H__
#define __GGML_FSMN_H__

#include "ggml_module.h"

#define FSMN_MAX_NODES 8192
#define VAD_CHUNK_SIZE 800

struct fsmn_model_t
{
    Linear in_linear1;
    Linear in_linear2;

    std::vector<FSMNBlock> encoders;

    Linear out_linear1;
    Linear out_linear2;
};

struct fsmn_state_t
{
    ggml_backend_sched_t sched = nullptr;
    std::vector<uint8_t> meta;

    std::vector<ggml_tensor *>in_caches;

    int value;
};

struct fsmn_params_t
{
    int sample_rate;
    std::string window;
    int num_mels;
    int frame_length;
    int frame_shift;
    int lfr_m;
    int lfr_n;

    uint32_t n_encoder_layer;

    int32_t min_speech_duration_ms;
    int32_t max_speech_duration_ms;
    int32_t min_silence_duration_ms;


    uint32_t n_batch;
};



#endif //  __GGML_FSMN_H__
