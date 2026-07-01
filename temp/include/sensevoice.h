#ifndef __SENSEVOICE_H__
#define __SENSEVOICE_H__

#include <cstdint>
#include <map>
#include <set>
#include "ggml-cpu.h"
#include "ggml-cpp.h"
#include "gguf.h"
#include "nn.h"
#include "gguf_loader.h"
#include "ggml_module.h"
#include "silero.h"

static const std::map<std::string, std::pair<int, std::string>> g_lang = {
        { "auto",  { 0,  "auto",         } },
        { "zh",  { 3,  "chinese",         } },
        { "en",  { 4,  "english",          } },
        { "yue",  { 7,  "cantonese",         } },
        { "ja",  { 11,  "japanese",         } },
        { "ko",  { 12,  "korean",          } },
        { "nospeech",  { 13,  "nospeech",          } },
};


typedef struct sensevoice_context_params
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
    const char * language;

    int n_mels = 80;  // dim of mels
    std::string window = "hamming";
    int frame_length = 25; 
    int frame_shift = 10; 
    int lfr_m = 7;
    int lfr_n = 6;
    int ftype = 1;
    float eps = 1e-5f;
    int n_audio_ctx = 1600;

    int n_batch;

    int32_t language_id = 0;

    //to do
    bool use_itn = true;
    bool flash_attn = true;

    //to do kv
    ggml_type wtype = ggml_type::GGML_TYPE_F16;  // weight type (FP32 / FP16 / QX)
    ggml_type itype =
            ggml_type::GGML_TYPE_F16;  // intermediate type (FP32 or FP16)

    nn_inference_buffer_policy_e inference_buffer_policy;

} sensevoice_context_params_t;

struct sensevoice_kv_cell {
    int32_t pos = -1; 

    std::set<int32_t> seq_id;

    bool has_seq_id(const int32_t & id) const {
        return seq_id.find(id) != seq_id.end();
    }   
};


struct sensevoice_kv_cache{
    uint32_t head = 0;
    uint32_t size = 0;

    uint32_t n = 0;

    std::vector<sensevoice_kv_cell> cells;

    ggml_tensor *k;
    ggml_tensor *v;

    ggml_context *ctx = nullptr;

    ggml_backend_buffer_t buffer = nullptr;
};




// to do merge
struct sensevoice_layer_encoder : Module
{
    // encoder_attn.linear_out.weight
    struct ggml_tensor *e_attn_ln_out_w;
    struct ggml_tensor *e_attn_ln_out_b;

    // encoder.self_attn.linear_q_k_v.weight
    struct ggml_tensor *e_attn_ln_q_w;
    struct ggml_tensor *e_attn_ln_q_b;

    struct ggml_tensor *e_attn_ln_k_w;
    struct ggml_tensor *e_attn_ln_k_b;

    struct ggml_tensor *e_attn_ln_v_w;
    struct ggml_tensor *e_attn_ln_v_b;

    // encoder.self_attn.fsmn_block.weight
    struct ggml_tensor *e_attn_fsmn_w;

    // encoder.feed_forward.w_1.weight
    struct ggml_tensor *e_mlp_w1;
    struct ggml_tensor *e_mlp_b1;

    // encoder.feed_forward.w_2.weight
    struct ggml_tensor *e_mlp_w2;
    struct ggml_tensor *e_mlp_b2;

    // encoder.norm1.weight
    struct ggml_tensor *e_norm_w1;
    struct ggml_tensor *e_norm_b1;

    // encoder.norm2.weight
    struct ggml_tensor *e_norm_w2;
    struct ggml_tensor *e_norm_b2;

    void OnLoad(const gguf_loader& loader, const std::string& prefix);

    ggml_tensor *build_cgraph(ggml_context* ctx, ggml_cgraph* gf, ggml_tensor* x, sensevoice_context_params_t &params, sensevoice_kv_cache &kv_cache, bool user_flash_attn) const;
};

struct sensevoice_encoder : Module
{
#if 0
    int n_encoder_hidden_state;
    int n_encoder_linear_units;
    int n_encoder_attention_heads;
    int n_encoder_layers;
    int n_tp_encoder_layers;
#endif

    int output_size;
    int linear_units;
    int attention_heads;
    int num_blocks;
    int tp_blocks;

    sensevoice_layer_encoder encoder0;
    std::vector<sensevoice_layer_encoder> encoders_layer;
    std::vector<sensevoice_layer_encoder> tp_encoders_layer;

    // encoder.tp_norm.weight
    ggml_tensor *e_tp_norm_w;
    ggml_tensor *e_tp_norm_b;

    // encoder.after_norm.weight
    ggml_tensor *e_after_norm_w;
    ggml_tensor *e_after_norm_b;

    void OnLoad(const gguf_loader& loader, const std::string& prefix);
    ggml_tensor *build_cgraph(ggml_context* ctx, ggml_cgraph* gf, ggml_tensor* x, sensevoice_context_params_t &params, sensevoice_kv_cache &kv_cache, bool user_flash_attn) const;
};


struct sensevoice_embedding : Module
{
    ggml_tensor *weight;

    void OnLoad(const gguf_loader& loader, const std::string& prefix);
};


struct ctc_linear : Module
{
    ggml_tensor *weight;
    ggml_tensor *bias;

    void OnLoad(const gguf_loader& loader, const std::string& prefix);
};

struct sensevoice_model 
{
    sensevoice_model(ggml_backend_t backend, const sensevoice_context_params_t& params);
    ~sensevoice_model();

    sensevoice_embedding embedding;

	sensevoice_encoder encoder;

    //decoder


    ctc_linear output;

    sensevoice_kv_cache kv_cache;

    int init_sensevoice_kv_cache();

    void empty_buffer_cache();

    void load(gguf_loader &loader);

    void build_cgraph();

    int inference();


    ggml_context_ptr ctx;
    ggml_context_ptr ctx0;

    ggml_backend_ptr backend;
    ggml_backend_ptr cpu_backend;
    ggml_backend_buffer_ptr buffer;
    ggml_backend_sched_ptr sched;

    ggml_cgraph* gf; 

    uint32_t sample_rate;

    ggml_backend_op_capabilities op_caps;

    ggml_status status;

    ggml_backend_buffer_ptr kv_buffer;

    sensevoice_context_params_t params;

    std::vector<uint8_t>meta;

    ggml_tensor *inputs[2];
};

#if 0
struct sensevoice_context : sensevoice_model
{
    int32_t language_id = 0;

    sensevoice_context(const sensevoice_context_params_t& params, ggml_backend_t backend) :
        sensevoice_model(backend, params) {}

};

typedef struct sensevoice_context *sensevoice_context_t;

void init_sensevoice_backend();
void sensevoice_default_params(sensevoice_context_params_t *params);

sensevoice_context_t load_sensevoice_model(const char *filename,
                                const sensevoice_context_params_t params,
                                ggml_backend_t backend,
                                uint32_t n_threads,
                                uint32_t reserved);
#endif
struct sensevoice_context : sensevoice_model
{
    sensevoice_context(const sensevoice_context_params_t& params, ggml_backend_t backend) :
        sensevoice_model(backend, params) {}

    // quant
    //bool llm_job();
};


typedef struct sensevoice_context *sensevoice_context_t;



void init_sensevoice_backend();
void sensevoice_default_params(sensevoice_context_params_t *params);

sensevoice_context_t load_sensevoice_model(const char *filename,
                                const sensevoice_context_params_t params,
                                ggml_backend_t backend,
                                uint32_t n_threads,
                                uint32_t reserved);


#endif //  __SENSEVOICE_H__
