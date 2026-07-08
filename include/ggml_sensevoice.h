#ifndef __GGML_SENSEVOICE_H__
#define __GGML_SENSEVOICE_H__

#include "ggml_module.h"

struct sensevoice_model_t
{
    ggml_context *ctx;
    ggml_backend_buffer_t buf_weights;

    SenseVoiceEncoderSmall encoder;

    CTC ctc;
};


struct sensevoice_state_t
{
    ggml_backend_sched_t sched = nullptr;

    ggml_tensor *feature;
    
    ggml_context *ctx_build;
    void *build_buf;
    int build_buf_size;

    ggml_context *ctx_cache;
    ggml_backend_buffer_t buf_cache;

};


struct sensevoice_params_t
{
    int n_vocab = 25055;                // number of vocab
    int n_max_audio_length = 20000;    //
    int n_encoder_hidden_state = 512;  // dim of hidden state
    int n_encoder_linear_units = 2048;
    int n_encoder_attention_heads = 4;  // head of self attention
    int n_encoder_layers = 50;          // num block of encoder
    int n_tp_encoder_layers = 20;
    int n_encoder_0_norm_size = 560;
    int n_decoder_hidden_state = 512;
    int n_decoder_linear_units = 2048;
    int n_decoder_attention_heads = 4;
    int n_decoder_layers = 14;
    int fsmn_kernel_size = 11;
    int n_vad_encoder_layers = 4;
    int n_predictor_dim = 512;
    float predictor_tail_threshold = 0.45;

    // for auto-detection, set to nullptr, "" or "auto"
    const char *language;

    int n_mels = 80;  // dim of mels
    char window[32];
    int frame_length = 25;
    int frame_shift = 10;
    int lfr_m = 7;
    int lfr_n = 6;
    int ftype = 1;
    float eps = 1e-5f;
    int n_audio_ctx = 1600;

    int n_batch;

    int32_t language_id = 0;

    bool use_itn = false;
    bool flash_attn = false;

    ggml_type wtype = ggml_type::GGML_TYPE_F16;  // weight type (FP32 / FP16 / QX)
    ggml_type itype =
            ggml_type::GGML_TYPE_F16;  // intermediate type (FP32 or FP16)

};

int load_sensevoice_model(struct ggml_handle_t *ggml_handle, const char *model_data, int model_size);
int sensevoice_inference(struct ggml_handle_t *ggml_handle, matrix_t **input_matrix,
                        matrix_t **output_matrix);

void unload_sensevoice_model(struct ggml_handle_t *ggml_handle);


#endif //  __GGML_SENSEVOICE_H__
