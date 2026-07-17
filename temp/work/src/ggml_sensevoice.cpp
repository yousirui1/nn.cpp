#include "base.h"
#include <map>
#include <string>
#include <fstream>
#include <sstream>
#include <set>

#include "gguf_loader.h"
#include "ggml_module.h"
#include "ggml_model.h"
#include "ggml_sensevoice.h"
#include "sensevoice_frontend.h"

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

struct sensevoice_layer_encoder_t {
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
};

struct sensevoice_encoder_t {
    ggml_type wtype = ggml_type::GGML_TYPE_F16;  // weight type (FP32 / FP16 / QX)
    ggml_type itype =
            ggml_type::GGML_TYPE_F16;  // intermediate type (FP32 or FP16)

    sensevoice_layer_encoder_t encoder0;

    std::vector<sensevoice_layer_encoder_t> encoders_layer;
    std::vector<sensevoice_layer_encoder_t> tp_encoders_layer;

    // encoder.tp_norm.weight
    struct ggml_tensor *e_tp_norm_w;
    struct ggml_tensor *e_tp_norm_b;

    // encoder.after_norm.weight
    struct ggml_tensor *e_after_norm_w;
    struct ggml_tensor *e_after_norm_b;
};

struct sensevoice_model_t 
{
    struct ggml_tensor *embedding;

    struct sensevoice_encoder_t *encoder;

    struct ggml_tensor *ctc_out_linear_weight;
    struct ggml_tensor *ctc_out_linear_bias;
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
    sensevoice_feature_t feature;
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


void get_sensevoice_default_params(struct sensevoice_params_t *params)
{
#if 0
    params->n_encoder_layer = 4;
    params->lstm_state_dim = 128;

    params->n_batch = 1;

    params->threshold      = 0.5f;
    params->neg_threshold = 0.35f;
    params->min_speech_duration_ms = 250;
    params->max_speech_duration_ms = 15000;
    params->min_silence_duration_ms = 100;
    params->speech_pad_ms = 30; 

    params->n_vocab = 25055;                // number of vocab
    params->n_max_audio_length = 20000;    //
    params->n_encoder_hidden_state = 512;  // dim of hidden state
    params->n_encoder_linear_units = 2048;
    params->n_encoder_attention_heads = 4;  // head of self attention
    params->n_encoder_layers = 50;          // num block of encoder
    params->n_tp_encoder_layers = 20;
    params->n_encoder_0_norm_size = 560;
    params->n_decoder_hidden_state = 512;
    params->n_decoder_linear_units = 2048;
    params->n_decoder_attention_heads = 4;
    params->n_decoder_layers = 14; 
    params->fsmn_kernel_size = 11; 
    params->n_vad_encoder_layers = 4;
    params->n_predictor_dim = 512;
    params->predictor_tail_threshold = 0.45;
    
    // for auto-detection, set to nullptr, "" or "auto"
    params->language = nullptr;
    
    params->n_mels = 80;  // dim of mels
    strncpy(params->window, "hamming", sizeof(params->window));
    params->frame_length = 25;
    params->frame_shift = 10;
    params->lfr_m = 7;
    params->lfr_n = 6;
    params->ftype = 1;
    params->eps = 1e-5f;
    params->n_audio_ctx = 1600;

    params->n_batch;

    params->language_id = 0;
    
    //to do
    params->use_itn = false;
    params->flash_attn = false;
    
    //to do kv
    params->wtype = ggml_type::GGML_TYPE_F16;  // weight type (FP32 / FP16 / QX)
    params->itype =
            ggml_type::GGML_TYPE_F16;  // intermediate type (FP32 or FP16)
#endif
};

static ggml_backend_buffer_type_t sensevoice_default_buffer_type(int use_gpu) {
    if (!use_gpu) {
        return ggml_backend_cpu_buffer_type();
    }
    for (size_t i = 0; i < ggml_backend_dev_count(); ++i) {
        ggml_backend_dev_t dev = ggml_backend_dev_get(i);
        if (ggml_backend_dev_type(dev) == GGML_BACKEND_DEVICE_TYPE_GPU) {
            LOG_INFO("%s: using device %s (%s)\n", __func__, ggml_backend_dev_name(dev), ggml_backend_dev_description(dev));
            return ggml_backend_dev_buffer_type(dev);
        }
    }

    return ggml_backend_cpu_buffer_type();
}

static bool init_sensevoice_sched_graph(struct sensevoice_sched_t &allocr, std::vector<ggml_backend_t> backends,
        std::function<struct ggml_cgraph *()> &&get_graph) {
    auto &sched = allocr.sched;
    auto &meta = allocr.meta;

    sched = ggml_backend_sched_new(backends.data(), nullptr, backends.size(), SENSEVOICE_MAX_NODE, false, true);

    meta.resize(ggml_tensor_overhead() * SENSEVOICE_MAX_NODE +
                ggml_graph_overhead());

    // since there are dependencies between the different graphs,
    // we need to allocate them instead of only reserving to get the correct compute buffer size
    if (!ggml_backend_sched_alloc_graph(sched, get_graph())) {
        // failed to allocate the compute buffer
        LOG_DEBUG("%s: failed to allocate the compute buffer\n", __func__);
        return false;
    }
    ggml_backend_sched_reset(sched);
    return true;
}

static bool init_sensevoice_kv_cache(struct sensevoice_kv_cache_t &cache, ggml_backend_t backend,
                    ggml_type wtype, int64_t n_text_state, int64_t n_text_layer, int n_ctx) 
{
    const int64_t n_mem = n_text_layer * n_ctx;
    const int64_t n_elements = n_text_state * n_mem;

    struct ggml_init_params params = {
            /*.mem_size   =*/2 * ggml_tensor_overhead(),
            /*.mem_buffer =*/nullptr,
            /*.no_alloc   =*/true,
    };

    cache.head = 0;
    cache.size = n_ctx;

    cache.cells.clear();
    cache.cells.resize(n_ctx);

    cache.ctx = ggml_init(params);

    if (!cache.ctx) {
        LOG_ERROR("%s: failed to allocate memory for the kv cache context\n", __func__);
        return false;
    }

    cache.k = ggml_new_tensor_1d(cache.ctx, wtype, n_elements);
    cache.v = ggml_new_tensor_1d(cache.ctx, wtype, n_elements);

    cache.buffer = ggml_backend_alloc_ctx_tensors(cache.ctx, backend);
    if (!cache.buffer) {
        LOG_ERROR("%s: failed to allocate memory for the kv cache\n", __func__);
        return false;
    }

    ggml_backend_buffer_clear(cache.buffer, 0);
    return true;
}




// faster matrix multiplications for tensors that do not have dimension 0 divisible by "pad"
// the idea is to represent the original matrix multiplication:
//
//   Z = X @ Y
//
// with the sum of two matrix multiplications:
//
//   Z = (X_0 @ Y_0) + (X_1 @ Y_1)
//
// here X_0 and Y_0 are views of X and Y that have dimension 0 divisible by "pad"
// and X_1 and Y_1 are the remaining views. X_1 and Y_1 end up being small matrices that can be processed with more
// general-purpose kernels
//
static struct ggml_tensor * ggml_mul_mat_pad(struct ggml_context * ctx, struct ggml_tensor * x, struct ggml_tensor * y, int pad = 32) {
    // use padding only if dimension 0 is at least 8 times larger than the padding
    // else we won't get much benefit from the optimization
    const int n_pad_req = 8;

    if (x->ne[0] % pad == 0 || x->ne[0] / pad < n_pad_req) {
        return ggml_mul_mat(ctx, x, y);
    }

    struct ggml_tensor * x_0 = ggml_view_3d(ctx, x, (x->ne[0]/pad)*pad, x->ne[1], x->ne[2], x->nb[1], x->nb[2], 0);
    struct ggml_tensor * x_1 = ggml_view_3d(ctx, x,  x->ne[0]%pad,      x->ne[1], x->ne[2], x->nb[1], x->nb[2], x_0->ne[0]*x_0->nb[0]);

    struct ggml_tensor * y_0 = ggml_view_3d(ctx, y, (y->ne[0]/pad)*pad, y->ne[1], y->ne[2], y->nb[1], y->nb[2], 0);
    struct ggml_tensor * y_1 = ggml_view_3d(ctx, y,  y->ne[0]%pad,      y->ne[1], y->ne[2], y->nb[1], y->nb[2], y_0->ne[0]*y_0->nb[0]);

    return ggml_add(ctx,
                    ggml_mul_mat(ctx, x_0, y_0),
                    ggml_mul_mat(ctx, x_1, y_1));
}

// copy from whisper.cpp
// TODO: CUDA is currently broken - seems ggml_mul_mat does not handle views correctly
#if defined(GGML_USE_METAL)
#define ggml_mul_mat ggml_mul_mat_pad
#endif

static struct ggml_tensor *encoder_layer_sanm_forward(const sensevoice_params_t params,
                                               sensevoice_state_t *state,
                                               ggml_context *ctx0,
                                               ggml_tensor *cur,
                                               sensevoice_layer_encoder_t &layer,
                                               ggml_cgraph *gf,
                                               bool user_flash_attn){

    const int n_state = params.n_encoder_hidden_state;
    const int n_head = params.n_encoder_attention_heads;

    struct ggml_tensor *residual = nullptr;

    if (layer.e_norm_w1->ne[0] == layer.e_norm_w2->ne[0]) {
        residual = ggml_cpy(
                ctx0, cur,
                ggml_new_tensor_3d(ctx0, cur->type, cur->ne[0], cur->ne[1], cur->ne[2]));
    }

    {
        // layer norm
        // cur = ln_0_w*cur + ln_0_b
#ifdef GGML_CUDA
        int32_t dim_size = cur->ne[0];
        if (sctx.params.use_gpu && dim_size % WARP_SIZE) {
            int32_t pad_size = WARP_SIZE - (dim_size % WARP_SIZE);
            ggml_tensor *mean = ggml_mean(ctx0, cur);
            cur = ggml_sub(ctx0, cur, mean);
            ggml_tensor *sigma = ggml_mul(ctx0, cur, cur);
            sigma = ggml_sum_rows(ctx0, sigma);
            cur = ggml_scale(ctx0, ggml_div(ctx0, cur, ggml_sqrt(ctx0, sigma)), sqrt(dim_size));
            // cur = ggml_cont(ctx0, ggml_pad(ctx0, cur, pad_size, 0, 0, 0));
            // cur = ggml_norm(ctx0, cur, params.eps);
            // cur = ggml_cont(ctx0, ggml_view_4d(ctx0, cur, dim_size, cur->ne[1], cur->ne[2], cur->ne[3], cur->nb[1], cur->nb[2], cur->nb[3], 0));
            // cur = ggml_scale(ctx0, cur, sqrt(float(dim_size) / (dim_size + pad_size)));
        }else{
            cur = ggml_norm(ctx0, cur, params.eps);
        }
#else
        cur = ggml_norm(ctx0, cur, params.eps);
#endif
        cur = ggml_add(ctx0, ggml_mul(ctx0, cur, layer.e_norm_w1), layer.e_norm_b1);
    }

    // self attention
    {
        // self attention linear qkv
        //      cur = ggml_transpose(ctx0, cur);
        // split qkv into separate tensors
        // q, k, v = torch.split(q_k_v, int(self.h * self.d_k), dim=-1)
        //  ref:
        //  https://github.com/alibaba-damo-academy/FunASR/blob/main/funasr/modules/attention.py#L391-L396
        struct ggml_tensor *Q;
        struct ggml_tensor *Q_h;
        struct ggml_tensor *K;
        struct ggml_tensor *K_h;
        struct ggml_tensor *V;
        struct ggml_tensor *V_h;

        int n_ctx = cur->ne[1];
        int n_batch = cur->ne[2];
        Q = ggml_add(ctx0,
                     ggml_mul_mat_pad(ctx0, layer.e_attn_ln_q_w, cur),
                     layer.e_attn_ln_q_b);

        Q_h = ggml_reshape_4d(ctx0, Q, n_state / n_head, n_head, n_ctx, n_batch);
        Q_h = ggml_permute(ctx0, Q_h,  0, 2, 1, 3);
        Q_h = ggml_cont(ctx0, Q_h);

        ggml_set_name(Q_h, "attention_Q");

        K = ggml_add(ctx0,
                     ggml_mul_mat(ctx0, layer.e_attn_ln_k_w, cur),
                     layer.e_attn_ln_k_b);

        K_h = ggml_reshape_4d(ctx0, K, n_state / n_head, n_head, n_ctx, n_batch);
        K_h = ggml_permute(ctx0, K_h, 0, 2, 1, 3);
        K_h = ggml_cont(ctx0, K_h);
        //      K = ggml_reshape_3d(ctx0, K, n_state, n_ctx, n_head);
        ggml_set_name(K_h, "attention_K");

        V = ggml_add(ctx0,
                     ggml_mul_mat(ctx0, layer.e_attn_ln_v_w, cur),
                     layer.e_attn_ln_v_b);
        ggml_set_name(V, "attention_V");

        V_h = ggml_reshape_4d(ctx0, V, n_state / n_head, n_head, n_ctx, n_batch);
        V_h = ggml_permute(ctx0, V_h, 0, 2, 1, 3);
        V_h = ggml_cont(ctx0, V_h);

        // fsmn forward with V
        int padding = (params.fsmn_kernel_size - 1) / 2;


        struct ggml_tensor *fsmn_memory = nullptr;
        // conv depth wise
        {
            {
                // implement conv depth wise with groups=input_channel implement
                // same in pytorch : F.conv1d(input, weight, bias=None, stride=1, padding=1, dilation=1, groups=n_state)
                struct ggml_tensor * a = layer.e_attn_fsmn_w;
                struct ggml_tensor * b = ggml_cont(ctx0, ggml_transpose(ctx0, V));
                // Process each batch separately and concatenate results
                // for (int i = 0; i < b->ne[2]; i++) {
                //     // View for current batch
                //     struct ggml_tensor *b_batch = ggml_view_3d(ctx0, b, b->ne[0], b->ne[1], 1, b->nb[1], b->nb[2], i * b->nb[2]);
                //     struct ggml_tensor *im2col = ggml_im2col(ctx0, a, ggml_reshape_4d(ctx0, b_batch, b_batch->ne[0], 1, b_batch->ne[1], b_batch->ne[2] * b_batch->ne[3]), 1, 0, padding, 0, 1, 0, false, GGML_TYPE_F32);
                //     struct ggml_tensor * result = ggml_mul_mat(ctx0, a, im2col);
                //     struct ggml_tensor * fsmn_memory_batch = ggml_reshape_3d(ctx0, result, im2col->ne[1], b_batch->ne[1], b_batch->ne[2]);
                //     if (fsmn_memory == nullptr) {
                //         fsmn_memory = fsmn_memory_batch;
                //     } else {
                //         fsmn_memory = ggml_concat(ctx0, fsmn_memory, fsmn_memory_batch, 2);
                //     }
                // }
                struct ggml_tensor * im2col = ggml_im2col(ctx0, a, ggml_reshape_4d(ctx0, b, b->ne[0], 1, b->ne[1] * b->ne[2], b->ne[3]), 1, 0, padding, 0, 1, 0, false, GGML_TYPE_F32);
                im2col = ggml_reshape_4d(ctx0, im2col, im2col->ne[0], im2col->ne[1], im2col->ne[2] / n_batch, n_batch);
                a = ggml_repeat(ctx0, ggml_cast(ctx0, a, GGML_TYPE_F32), ggml_new_tensor_4d(ctx0, GGML_TYPE_F16, a->ne[0], a->ne[1], a->ne[2], n_batch));
                struct ggml_tensor * result = ggml_mul_mat(ctx0, a, im2col);
                fsmn_memory = ggml_reshape_3d(ctx0, result, im2col->ne[1], im2col->ne[2], im2col->ne[3]);
            }
            fsmn_memory = ggml_cont(ctx0, ggml_transpose(ctx0, fsmn_memory));
            fsmn_memory = ggml_add(ctx0, fsmn_memory, V);
            ggml_set_name(fsmn_memory, "fsmn_memory");
        }

        float KQscale = 1.0f / sqrtf(float(n_state) / n_head);

        if(user_flash_attn){
            const int n_ctx_pad = GGML_PAD(n_ctx, 256);
            const int n_state_head = n_state / n_head;

            ggml_build_forward_expand(gf, ggml_cpy(ctx0, K, ggml_view_1d(ctx0, state->kv_pad.k, n_ctx*n_state*n_batch, 0)));
            ggml_build_forward_expand(gf, ggml_cpy(ctx0, V, ggml_view_1d(ctx0, state->kv_pad.v, n_ctx*n_state*n_batch, 0)));

            struct ggml_tensor * K =
                    ggml_view_4d(ctx0, state->kv_pad.k,
                                 n_state_head, n_ctx_pad, n_head, n_batch,
                                 ggml_element_size(state->kv_pad.k)*n_state,
                                 ggml_element_size(state->kv_pad.k)*n_state_head,
                                 ggml_element_size(state->kv_pad.k)*n_state*n_ctx_pad,
                                 0);

            struct ggml_tensor * V =
                    ggml_view_4d(ctx0, state->kv_pad.v,
                                 n_state_head, n_ctx_pad, n_head, n_batch,
                                 ggml_element_size(state->kv_pad.v)*n_state,
                                 ggml_element_size(state->kv_pad.v)*n_state_head,
                                 ggml_element_size(state->kv_pad.v)*n_state*n_ctx_pad,
                                 0);
            ggml_tensor *KQV = ggml_flash_attn_ext(ctx0, Q_h, K, V, nullptr, KQscale, 0.0f, 0.0f);
            cur = ggml_reshape_3d(ctx0, KQV, n_state, n_ctx, n_batch);
        } else{
            // K * Q
            struct ggml_tensor *KQ = ggml_mul_mat(ctx0, K_h, Q_h);

            struct ggml_tensor *KQ_soft_max = ggml_soft_max_ext(ctx0, KQ, nullptr, KQscale, 0.0f);


            ggml_tensor *KQV = ggml_mul_mat(
                    ctx0, ggml_cont(ctx0, ggml_transpose(ctx0, V_h)), KQ_soft_max);
            struct ggml_tensor *KQV_merged = ggml_permute(ctx0, KQV, 0, 2, 1, 3);
            cur = ggml_cpy(ctx0,
                           KQV_merged,
                           ggml_new_tensor_3d(ctx0, GGML_TYPE_F32, n_state, n_ctx, n_batch));
        }

        cur = ggml_add(ctx0, ggml_mul_mat(ctx0, layer.e_attn_ln_out_w, cur),
                       layer.e_attn_ln_out_b);
        ggml_set_name(cur, "attention_out");

        cur = ggml_add(ctx0, cur, fsmn_memory);

        if (layer.e_norm_w1->ne[0] == layer.e_norm_w2->ne[0]) {
            cur = ggml_add(ctx0, cur, residual);
        }
    }

    residual = ggml_cpy(
            ctx0, cur,
            ggml_new_tensor_3d(ctx0, GGML_TYPE_F32, cur->ne[0], cur->ne[1], cur->ne[2]));
    {
        // layer norm after attention
        // cur = ln_0_w*cur + ln_0_b
        cur = ggml_norm(ctx0, cur, params.eps);
        cur = ggml_add(ctx0, ggml_mul(ctx0, cur, layer.e_norm_w2), layer.e_norm_b2);
    }

    {
        // position-wise feed forward layer
        cur = ggml_add(ctx0, ggml_mul_mat(ctx0, layer.e_mlp_w1, cur),
                       layer.e_mlp_b1);
        cur = ggml_relu(ctx0, cur);
        cur = ggml_add(ctx0, ggml_mul_mat(ctx0, layer.e_mlp_w2, cur),
                       layer.e_mlp_b2);
    }
    // residual after position wise feed forward
    cur = ggml_add(ctx0, cur, residual);
    return cur;

}

ggml_cgraph *sensevoice_build_cgraph_encoder(struct sensevoice_model_t *model, struct sensevoice_state_t *state, struct sensevoice_params_t *hparams)
{
    bool flash_attn = hparams->flash_attn;
    struct ggml_init_params params = {
            /*.mem_size   =*/state->sched_encode.meta.size(),
            /*.mem_buffer =*/state->sched_encode.meta.data(),
            /*.no_alloc   =*/true,
    };

    struct ggml_context *ctx0 = ggml_init(params);

    ggml_cgraph *gf = ggml_new_graph_custom(ctx0, SENSEVOICE_ENCODER_MAX_NODES, false);

    struct ggml_tensor *feature = state->feature.tensor;
    ggml_set_name(feature, "feats");
    ggml_set_input(feature);

    struct ggml_tensor *embedding = ggml_new_tensor_2d(ctx0, GGML_TYPE_I32, 4, 1);
    ggml_set_name(embedding, "embedding");
    ggml_set_input(embedding);

    embedding = ggml_get_rows(ctx0, model->embedding, embedding);
    embedding = ggml_repeat(ctx0, embedding, ggml_new_tensor_3d(ctx0, GGML_TYPE_I32, embedding->ne[0], embedding->ne[1], feature->ne[2]));

    struct ggml_tensor *cur = ggml_concat(ctx0, embedding, feature, 1);

    cur = ggml_scale(ctx0, cur, sqrtf(hparams->n_encoder_hidden_state));

    // implement encoder small forward graph
    //ref: https://github.com/modelscope/FunASR/blob/b7b4a83c18277a7022124cad790c08ae703b7a2d/funasr/models/sense_voice/model.py#L558-L583
    // [x] 1. sinusoidal position
    // [x] 2. encoders0
    // [x] 3. encoders
    // [x] 4. tp_encoders
    // [x] 5. tp_norm
    ggml_tensor *position = ggml_new_tensor_3d(ctx0, cur->type, cur->ne[0], cur->ne[1], cur->ne[2]);
    ggml_set_name(position, "position");
    ggml_set_input(position);

    cur = ggml_add(ctx0, position, cur);

    // encoders0 forward
    cur = encoder_layer_sanm_forward(*hparams, state, ctx0, cur, model->encoder->encoder0, gf, flash_attn);

    // encoders forward
    for (int i=0; i < hparams->n_encoder_layers - 1; i++){
        cur = encoder_layer_sanm_forward(*hparams, state, ctx0, cur, model->encoder->encoders_layer[i], gf, flash_attn);
    }

    {
        // after encoder norm
        cur = ggml_norm(ctx0, cur, hparams->eps);
        cur = ggml_add(ctx0, ggml_mul(ctx0,
                                      cur,
                                      model->encoder->e_after_norm_w),
                       model->encoder->e_after_norm_b);
    }

    // tp encoders forward
    for (int i=0; i < hparams->n_tp_encoder_layers; i++){
        cur = encoder_layer_sanm_forward(*hparams,  state, ctx0, cur, model->encoder->tp_encoders_layer[i], gf, flash_attn);
    }

    {
        // tp encoder norm
        cur = ggml_norm(ctx0, cur, hparams->eps);
        cur = ggml_add(ctx0, ggml_mul(ctx0,
                                      cur,
                                      model->encoder->e_tp_norm_w),
                       model->encoder->e_tp_norm_b);
    }


    ggml_build_forward_expand(gf, cur);

    ggml_set_name(cur, "encoder_out");
    ggml_set_output(cur);
    state->encoder_out = cur;
    ggml_free(ctx0);
    return gf;
}



ggml_cgraph *sensevoice_build_cgraph_ctc_decoder(struct sensevoice_model_t *model, struct sensevoice_state_t *state)
{
    struct ggml_init_params params = {
            /*.mem_size   =*/state->sched_decode.meta.size(),
            /*.mem_buffer =*/state->sched_decode.meta.data(),
            /*.no_alloc   =*/true,
    };

    struct ggml_context *ctx0 = ggml_init(params);

    ggml_cgraph *gf = ggml_new_graph_custom(ctx0, SENSEVOICE_DECODER_MAX_NODES, false);

    ggml_tensor *encoder_out = ggml_new_tensor_3d(ctx0, state->encoder_out->type,
                                                  state->encoder_out->ne[0], state->encoder_out->ne[1],
                                                  state->encoder_out->ne[2]);
    ggml_set_name(encoder_out, "encoder_out");
    ggml_set_input(encoder_out);

    ggml_tensor *cur;
    {
        // Reshape encoder_out to merge batch and time dimensions
        cur = ggml_reshape_2d(ctx0, encoder_out, encoder_out->ne[0], encoder_out->ne[1] * encoder_out->ne[2]);
        cur = ggml_mul_mat(ctx0, model->ctc_out_linear_weight, cur);
        cur = ggml_add(ctx0, cur, model->ctc_out_linear_bias);
        // Reshape back to 3D
        cur = ggml_reshape_3d(ctx0, cur, cur->ne[0], encoder_out->ne[1], encoder_out->ne[2]);
    }
    ggml_tensor * probs = ggml_soft_max(ctx0, cur);
    probs = ggml_reshape_2d(ctx0, probs, probs->ne[0], probs->ne[1] * probs->ne[2] * probs->ne[3]);
    ggml_tensor * argmax_logit = ggml_argmax(ctx0, probs);
    argmax_logit = ggml_reshape_3d(ctx0, argmax_logit, cur->ne[1], cur->ne[2], cur->ne[3]);
    ggml_set_output(probs);
    ggml_set_output(argmax_logit);
    ggml_build_forward_expand(gf, argmax_logit);
    ggml_free(ctx0);
    return gf;
}

static bool set_sensevoice_encoder_layer_sanm(std::vector<sensevoice_layer_encoder_t> &encoder,
                                        std::map<std::string,
                                        struct ggml_tensor *> &tensors,
                                        int n_encoder_layers,
                                        const std::string &prefix){

    for (int i = 0; i < n_encoder_layers; ++i) {
        auto layer = &encoder[i];
        // map by name
        layer->e_attn_ln_out_w =
                tensors["encoder." + prefix + "." + std::to_string(i) +
                        ".self_attn.linear_out.weight"];
        layer->e_attn_ln_out_b =
                tensors["encoder." + prefix + "." + std::to_string(i) +
                        ".self_attn.linear_out.bias"];

        layer->e_attn_ln_q_w =
                tensors["encoder." + prefix + "." + std::to_string(i) +
                        ".self_attn.linear_q.weight"];
        layer->e_attn_ln_q_b =
                tensors["encoder." + prefix + "." + std::to_string(i) +
                        ".self_attn.linear_q.bias"];

        layer->e_attn_ln_k_w =
                tensors["encoder." + prefix + "." + std::to_string(i) +
                        ".self_attn.linear_k.weight"];
        layer->e_attn_ln_k_b =
                tensors["encoder." + prefix + "." + std::to_string(i) +
                        ".self_attn.linear_k.bias"];

        layer->e_attn_ln_v_w =
                tensors["encoder." + prefix + "." + std::to_string(i) +
                        ".self_attn.linear_v.weight"];
        layer->e_attn_ln_v_b =
                tensors["encoder." + prefix + "." + std::to_string(i) +
                        ".self_attn.linear_v.bias"];

        layer->e_attn_fsmn_w =
                tensors["encoder." + prefix + "." + std::to_string(i) +
                        ".self_attn.fsmn_block.weight"];

        layer->e_mlp_w1 =
                tensors["encoder." + prefix + "." + std::to_string(i) +
                        ".feed_forward.w_1.weight"];
        layer->e_mlp_b1 =
                tensors["encoder." + prefix + "." + std::to_string(i) +
                        ".feed_forward.w_1.bias"];

        layer->e_mlp_w2 =
                tensors["encoder." + prefix + "." + std::to_string(i) +
                        ".feed_forward.w_2.weight"];
        layer->e_mlp_b2 =
                tensors["encoder." + prefix + "." + std::to_string(i) +
                        ".feed_forward.w_2.bias"];

        layer->e_norm_w1 =
                tensors["encoder." + prefix + "." + std::to_string(i) +
                        ".norm1.weight"];
        layer->e_norm_b1 =
                tensors["encoder." + prefix + "." + std::to_string(i) +
                        ".norm1.bias"];

        layer->e_norm_w2 =
                tensors["encoder." + prefix + "." + std::to_string(i) +
                        ".norm2.weight"];
        layer->e_norm_b2 =
                tensors["encoder." + prefix + "." + std::to_string(i) +
                        ".norm2.bias"];
    }
    return true;
}



int load_sensevoice_model(struct ggml_handle_t *ggml_handle, const char *model_path,
                        shape_t *input_shape, int *in_nodes,shape_t *output_shape, int *out_nodes)
{
    struct ggml_context *ctx;
    ggml_backend_buffer_t buffer = nullptr;

    ggml_backend_t backend;
    ggml_backend_t cpu_backend;
    std::map<std::string, struct ggml_tensor *> tensors;

    struct sensevoice_params_t *params = new sensevoice_params_t;
    if(NULL == params)
    {
        LOG_ERROR("sensevoice params error %s", strerror(errno));
        return ERROR;
    }

    get_sensevoice_default_params(params);

    struct sensevoice_model_t *model = new sensevoice_model_t; 
    if(NULL == model)
    {
        LOG_ERROR("sensevoice malloc error %s", strerror(errno));
        delete params;
        return ERROR;
    }

    struct sensevoice_state_t *state = new sensevoice_state_t; 
    if(NULL == model)
    {
        LOG_ERROR("sensevoice state error %s", strerror(errno));
        delete model;
        delete params;
        return ERROR;
    }

    backend = ggml_backend_init_best();
    cpu_backend = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr);

    state->backends.push_back(backend);
    state->backends.push_back(cpu_backend);

    struct gguf_init_params gguf_params = { 
            /*.no_alloc = */ true,
            /*.ctx      = */ &ctx,
    };

	struct gguf_context *gguf_ctx = gguf_init_from_file(model_path, gguf_params);

    // load vocab
    {
        std::string word;

        const int token_idx = gguf_find_key(gguf_ctx, "tokenizer.ggml.tokens");
        const int n_vocab = gguf_get_arr_n(gguf_ctx, token_idx);

        if (n_vocab != params->n_vocab) 
		{
            LOG_ERROR("vocabulary loaded from model file error - vocabulary size is "
                    "%d, but got %d .",
                    params->n_vocab, n_vocab);
        }
        std::vector<const char *> tokens;
        tokens.resize(n_vocab);
        for (int i = 0; i < n_vocab; i++) {
            word = gguf_get_arr_str(gguf_ctx, token_idx, i);
            state->vocab.token_to_id[word] = i;
            state->vocab.id_to_token[i] = word;
        }
        LOG_INFO("%s: vocab[%d] loaded", __func__, n_vocab);
    }

    buffer = ggml_backend_alloc_ctx_tensors_from_buft(ctx, sensevoice_default_buffer_type(1));
    if (!buffer) 
    {
        LOG_ERROR(" failed to allocate memory for the model");
        return ERROR;
    }

    size_t size_main = ggml_backend_buffer_get_size(buffer);
    LOG_INFO("%s: %s total size = %8.2f MB\n", __func__, ggml_backend_buffer_name(buffer), size_main / 1e6);

    int n_loaded = 0;

    // load weights
    {
        // host buffer for CUDA loading
        std::vector<uint8_t> read_buf;

        const int n_tensors = gguf_get_n_tensors(gguf_ctx);
        LOG_INFO("%s: n_tensors: %d\n", __func__, n_tensors);


        // model tensor sizing
        size_t buffer_size = 32 * 1024;// need some extra room??

        for (int i = 0; i < n_tensors; ++i) {
            const char *name = gguf_get_tensor_name(gguf_ctx, i);
            struct ggml_tensor *cur = ggml_get_tensor(ctx, name);
            size_t tensor_size = ggml_nbytes(cur);
            buffer_size += tensor_size;
        }


        // open model gguf file
        auto fin = std::ifstream(model_path, std::ios::binary);
        if (!fin) {
            fprintf(stderr, "cannot open model file for loading tensors\n");
            return false;
        }

        for (int i = 0; i < n_tensors; ++i) {
            const std::string name = gguf_get_tensor_name(gguf_ctx, i);
            struct ggml_tensor *cur = ggml_get_tensor(ctx, name.c_str());
            tensors[name] = cur;

            // seek to the tensor data in the file
            const size_t offset = gguf_get_data_offset(gguf_ctx) + gguf_get_tensor_offset(gguf_ctx, i);
            fin.seekg(offset, std::ios::beg);
            if (!fin) {
                fprintf(stderr, "%s: failed to seek for tensor %s\n", __func__, name.c_str());
                return false;
            }

            // read in data and copy to device if needed
            int num_bytes = ggml_nbytes(cur);
            if (ggml_backend_buffer_is_host(buffer)) {
                // for the CPU and Metal backend, we can read directly into the tensor
                fin.read(reinterpret_cast<char *>(cur->data), num_bytes);
            } else {
                // read into a temporary buffer first, then copy to device memory
                read_buf.resize(num_bytes);
                fin.read(reinterpret_cast<char *>(read_buf.data()), num_bytes);
                ggml_backend_tensor_set(cur, read_buf.data(), 0, num_bytes);
            }

            auto n_dim = ggml_n_dims(cur);
            std::stringstream shape;
            if (n_dim == 1)
                shape << cur->ne[0];
            else if (n_dim == 2)
                shape << cur->ne[0] << ',' << cur->ne[1];
            else if (n_dim == 3)
                shape << cur->ne[0] << ',' << cur->ne[1] << ',' << cur->ne[2];
            else
                shape << cur->ne[0] << ',' << cur->ne[1] << ',' << cur->ne[2] << ','
                      << cur->ne[3];
        }
        ggml_backend_buffer_set_usage(buffer, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
        gguf_free(gguf_ctx);
    }

    model->encoder = new sensevoice_encoder_t;

    // load encoder weights, multi layers of EncoderLayerSANM
    {
        model->embedding = tensors["embed.weight"];
        std::vector<sensevoice_layer_encoder_t> tmp_encoder0;
        tmp_encoder0.resize(1);

        //tmp_encoder0.push_back(model.encoder->encoder0);
        set_sensevoice_encoder_layer_sanm(tmp_encoder0, tensors, 1, "encoders0");

        model->encoder->encoder0 = tmp_encoder0[0];

        model->encoder->encoders_layer.resize(params->n_encoder_layers - 1);
        set_sensevoice_encoder_layer_sanm(model->encoder->encoders_layer, tensors, params->n_encoder_layers - 1, "encoders");
        model->encoder->tp_encoders_layer.resize(params->n_tp_encoder_layers);
        set_sensevoice_encoder_layer_sanm(model->encoder->tp_encoders_layer, tensors, params->n_tp_encoder_layers, "tp_encoders");

        model->encoder->e_after_norm_w = tensors["encoder.after_norm.weight"];
        model->encoder->e_after_norm_b = tensors["encoder.after_norm.bias"];

        model->encoder->e_tp_norm_w = tensors["encoder.tp_norm.weight"];
        model->encoder->e_tp_norm_b = tensors["encoder.tp_norm.bias"];

        model->ctc_out_linear_weight = tensors["ctc.ctc_lo.weight"];
        model->ctc_out_linear_bias = tensors["ctc.ctc_lo.bias"];
    }


    if (!init_sensevoice_kv_cache(state->kv_pad, state->backends[0], params->itype,
                                   params->n_encoder_hidden_state,
                                   1,
                                   GGML_PAD(params->n_audio_ctx, 256))) {
        LOG_ERROR("sense_voice_kv_cache_init() failed for self-attention cache\n");
        //sensevoice_free_state(state);
        return ERROR;
    }

    // set input
    {
        // init features
        state->feature.n_len = SENSEVOICE_CHUNK_SIZE;
        state->feature.ctx = ggml_init({ggml_tensor_overhead(), nullptr, true});
        state->feature.tensor = ggml_new_tensor_2d(state->feature.ctx,
                                                   GGML_TYPE_F32,
                                                   SENSEVOICE_FEATURES_DIM,
                                                   state->feature.n_len);
        state->feature.backend = backend;
    }

    // encoder allocator
    {
        bool ok = init_sensevoice_sched_graph(state->sched_encode, state->backends,
                [&]() { return sensevoice_build_cgraph_encoder(model, state, params); });

       if (!ok) 
       {
            LOG_ERROR("failed to init encode allocator");
            return ERROR;
        }
    }

    // decoder allocator
    {
        bool ok = init_sensevoice_sched_graph(
                state->sched_decode, state->backends,
                [&]() { return sensevoice_build_cgraph_ctc_decoder(model, state); });

        if (!ok) 
        {
            LOG_ERROR("failed to init encode allocator\n");
            return ERROR;
        }
    }

    ggml_handle->params = (void *)params;
    ggml_handle->model = (void *)model;
    ggml_handle->state = (void *)state;

    *in_nodes = 1;
    set_shape(&input_shape[0], TENSOR_FLOAT32, 1, -1);

    *out_nodes = 1;
    set_shape(&output_shape[0], TENSOR_INT8, 1, params->n_max_audio_length);

    return SUCCESS;
}


int sensevoice_encoder_inference(struct sensevoice_model_t *model, 
            sensevoice_params_t *hparams, sensevoice_state_t *state, const int n_threads)
{

    auto & sched = state->sched_encode.sched;
    ggml_cgraph *gf = sensevoice_build_cgraph_encoder(model, state, hparams);

    if (!ggml_backend_sched_alloc_graph(sched, gf)) 
    {
        // should never happen as we pre-allocate the memory
        LOG_ERROR("ggml_backend_sched_alloc_graph error");
        return ERROR;
    }

    // set the inputs
    {
        struct ggml_tensor *position = ggml_graph_get_tensor(gf, "position");
        struct ggml_tensor *embedding = ggml_graph_get_tensor(gf, "embedding");


        auto n_len = position->ne[1];
        auto dim = position->ne[0];
        auto n_batch = position->ne[2];
        std::vector<float> _position;
        _position.resize(n_len * dim * n_batch);
        // construct position embedding
        // sinusoidal position embedding
        // reference:
        // https://github.com/modelscope/FunASR/blob/45d7aa9004763684fb748ee17942ecba81042201/funasr/models/transformer/embedding.py#L392-L405
        // P_{k,i} = sin(k/10000^(2i/d))  0 < i < d/2
        // p_{k,j} = cos(k/10000^(2j/d))  d/2 < j < d

        for (int b = 0; b < n_batch; b++)
        for (int k = 1; k <= n_len; k++) {
            for (int i = 0; i < dim / 2; i++) {
            _position[b * n_len * dim + (k - 1) * dim + i] = sinf(k * pow(10000, -2.0 * i / dim));
            _position[b * n_len * dim + (k - 1) * dim + i + dim / 2] =
                    cosf(k * pow(10000, -2.0 * i / dim)); 
            }
        }


        ggml_backend_tensor_set(
                position, _position.data(), 0,
                ggml_nelements(position) * sizeof(float));

        int _embedding[4] = {hparams->language_id, 1, 2, hparams->use_itn ? 14 : 15};
        bool use_itn = false;
        //int _embedding[4] = {0, 1, 2, use_itn ? 14 : 15};
        ggml_backend_tensor_set(embedding, _embedding, 0, 4*sizeof(int));
    }
//  ggml_graph_dump_dot(gf, NULL, "sense-voice.dot");
//  ggml_backend_sched_set_eval_callback(sched, ctx.params.cb_eval, ctx.params.cb_eval_user_data);
    if (!ggml_graph_compute_helper(sched, gf, n_threads)) 
    {
        return ERROR;
    }
    struct ggml_tensor *position = ggml_graph_get_tensor(gf, "position");

    return SUCCESS;;
}

int sensevoice_decoder_ctc_inference(struct sensevoice_model_t *model, 
            sensevoice_state_t *state, const int n_threads)
{

    auto & sched = state->sched_decode.sched;
    ggml_cgraph *gf = sensevoice_build_cgraph_ctc_decoder(model, state);

    if (!ggml_backend_sched_alloc_graph(sched, gf)) 
    {
        // should never happen as we pre-allocate the memory
        LOG_ERROR("ggml_backend_sched_alloc_graph error");
        return ERROR;
    }

    // set the inputs
    {
        struct ggml_tensor *encoder_out = ggml_graph_get_tensor(gf, "encoder_out");
        ggml_backend_tensor_copy(state->encoder_out, encoder_out);
    }


    if (!ggml_graph_compute_helper(sched, gf, n_threads)) 
    {
        return ERROR;
    }

    ggml_tensor *argmax_logit = ggml_graph_node(gf, ggml_graph_n_nodes(gf) - 1);
    // TODO 临时处理，建议讨论后取其一
    if(state->result_all.empty()) 
    {
        state->ids.resize(argmax_logit->ne[0]);
        ggml_backend_tensor_get(argmax_logit, state->ids.data(), 0, sizeof(int) * argmax_logit->ne[0]);
    }
    else {
        const int32_t n_logits = argmax_logit->ne[0] * argmax_logit->ne[1];
        // Get the tensor data into a temporary buffer
        std::vector<int> temp_buffer(n_logits);
        ggml_backend_tensor_get(argmax_logit, temp_buffer.data(), 0, sizeof(int) * n_logits);
        for(int32_t i = 0; i < argmax_logit->ne[1]; i++)
        {
            int posL = i * argmax_logit->ne[0];
            state->result_all[state->segmentIDs[i]].tokens = std::vector<int>(temp_buffer.begin() + posL, temp_buffer.begin() + posL + argmax_logit->ne[0]);
        }
    }
    return SUCCESS;;
}

void sensevoice_print_output(struct sensevoice_state_t *state, sensevoice_vocab_t vocab,  bool need_prefix, std::string &output) 
{
    for (size_t i = (need_prefix ? 0 : 4); i < state->ids.size(); i++)
    {
        int id = state->ids[i];
        if (i > 0 && state->ids[i - 1] == state->ids[i])
            continue;
        if (id)
        {
            output += vocab.id_to_token[id];
        }
    }
}


int sensevoice_inference(struct ggml_handle_t *ggml_handle, matrix_t **input_matrix, matrix_t **output_matrix, int n_threads, void *param)
{
    struct sensevoice_params_t *params = (struct sensevoice_params_t *)ggml_handle->params;
    struct sensevoice_model_t *model = (struct sensevoice_model_t *)ggml_handle->model;
    struct sensevoice_state_t *state = (struct sensevoice_state_t *)ggml_handle->state;

    std::string output = "";

    //to do float->double
    std::vector<double>speech_segment;
    speech_segment.resize(input_matrix[0]->shape.size);
    for(int i = 0; i < input_matrix[0]->shape.size; i++)
    {
        speech_segment[i] = input_matrix[0]->data_fp[i];
    }

    sensevoice_pcm2feature_with_state(state->feature, speech_segment, false, n_threads); 

    sensevoice_encoder_inference(model, params, state, n_threads);
    sensevoice_decoder_ctc_inference(model, state, n_threads);

    sensevoice_print_output(state, state->vocab, false, output);

    if(output_matrix[0]->shape.size >= output.size())
    {
        memcpy(output_matrix[0]->data, output.c_str(), output.size());
        output_matrix[0]->shape.dims[0] = output.size();
    }
    
    return SUCCESS;
}


