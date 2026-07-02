#ifndef __GGML_SENSEVOICE_H__
#define __GGML_SENSEVOICE_H__

#include <map>
#include <string>
#include <fstream>
#include <sstream>
#include <set>

#include "ggml_module.hpp"



#define SENSEVOICE_MAX_NODE 8192
#define SENSEVOICE_CHUNK_SIZE 20
#define SENSEVOICE_FEATURES_DIM 560
#define SENSEVOICE_ENCODER_MAX_NODES 8129
#define SENSEVOICE_DECODER_MAX_NODES 16

static const std::map<std::string, std::pair<int, std::string>> g_lang = {
        { "auto",  { 0,  "auto",         } },
        { "zh",  { 3,  "chinese",         } },
        { "en",  { 4,  "english",          } },
        { "yue",  { 7,  "cantonese",         } },
        { "ja",  { 11,  "japanese",         } },
        { "ko",  { 12,  "korean",          } },
        { "nospeech",  { 13,  "nospeech",          } },
};

struct sensevoice_vocab_t
{
    using id = int32_t;
    using token = std::string;

    int n_vocab = 25055;

    std::map<token, id> token_to_id;
    std::map<id, token> id_to_token;

    id token_eot = 2;
    id token_sot = 1;
};

struct sensevoice_segment_t
{
    size_t t0;
    size_t t1;
    std::vector<int> tokens;     // 识别后的tokens
    std::vector<float> samples;  // 具体音频
    // std::vector<float>
    // bool speaker_turn_next;
};

struct sensevoice_sched_t {
    ggml_backend_sched_t sched = nullptr;
    std::vector<uint8_t> meta;
};

struct sensevoice_kv_cell_t {
    int32_t pos = -1;

    std::set<int32_t> seq_id;

    bool has_seq_id(const int32_t & id) const {
        return seq_id.find(id) != seq_id.end();
    }
};

struct sensevoice_kv_cache_t {
    uint32_t head = 0;
    uint32_t size = 0;

    // computed before each graph build
    uint32_t n = 0;

    std::vector<sensevoice_kv_cell_t> cells;

    struct ggml_tensor * k;
    struct ggml_tensor * v;

    struct ggml_context * ctx = nullptr;

    ggml_backend_buffer_t buffer = nullptr;
};

struct sensevoice_state_t
{
    //sensevoice_feature_t feature; //to do 
    sensevoice_vocab_t vocab;

    sensevoice_sched_t sched_encode;
    sensevoice_sched_t sched_decode;

    std::vector<ggml_backend_t> backends;

    sensevoice_kv_cache_t kv_pad;

    std::vector<sensevoice_segment_t> result_all;
    std::vector<int> ids;
    std::vector<size_t> segmentIDs;


    struct ggml_tensor *encoder_out;
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

    //nn_inference_buffer_policy_e inference_buffer_policy;
};



#endif //  __GGML_SENSEVOICE_H__
