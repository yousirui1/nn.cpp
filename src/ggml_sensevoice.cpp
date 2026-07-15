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

struct sensevoice_model_t 
{
    ggml_context *ctx;
    ggml_backend_buffer_t buf_weights;

    sensevoice_vocab_t vocab;
    struct ggml_tensor *embedding;

    SenseVoiceEncoderSmall encoder;

    CTC ctc;
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

    sensevoice_sched_t sched_encode;
    sensevoice_sched_t sched_decode;

	std::vector<ggml_backend_t> backends;

    sensevoice_kv_cache_t kv_pad;

    std::vector<sensevoice_segment_t> result_all;
    std::vector<int> ids;
    std::vector<size_t> segmentIDs;


    struct ggml_tensor *encoder_out;

    // Model weight buffer
    ggml_backend_buffer_t model_buffer = nullptr;
    struct ggml_context *model_ctx = nullptr;
};


void get_sensevoice_default_params(struct sensevoice_params_t *params)
{
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

    ggml_tensor *cur = model->encoder.build_cgraph(ctx0, feature, hparams->fsmn_kernel_size, 0);

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

    auto [probs, argmax_logit] = model->ctc.build_cgraph(ctx0, encoder_out);

    ggml_set_output(probs);
    ggml_set_output(argmax_logit);

    ggml_build_forward_expand(gf, argmax_logit);
    ggml_free(ctx0);
    return gf;
}


int load_sensevoice_tokenizer(gguf_metadata_loader& gguf_ctx, sensevoice_vocab_t &vocab)
{
    std::string word;

    const int token_idx = gguf_find_key(gguf_ctx, "tokenizer.ggml.tokens");
    const int n_vocab = gguf_get_arr_n(gguf_ctx, token_idx);

#if 0
    if (n_vocab != params->n_vocab) 
	{
        LOG_ERROR("vocabulary loaded from model file error - vocabulary size is "
                "%d, but got %d .",
                params->n_vocab, n_vocab);
    }
#endif
    std::vector<const char *> tokens;
    tokens.resize(n_vocab);
    for (int i = 0; i < n_vocab; i++) 
    {
        word = gguf_get_arr_str(gguf_ctx, token_idx, i);
        vocab.token_to_id[word] = i;
        vocab.id_to_token[i] = word;
    }
    LOG_INFO("%s: vocab[%d] loaded", __func__, n_vocab);
    return SUCCESS;
}


int load_sensevoice_weight(gguf_loader& loader, sensevoice_model_t *model, ggml_backend_t backend)
{
    std::string prefix = "encoder";
    model->ctx = ggml_init(ggml_init_params{ .mem_size = ggml_graph_overhead() * GGML_DEFAULT_GRAPH_SIZE, .no_alloc = true });

    model->encoder.onload(loader, prefix);
    model->ctc.onload(loader, "ctc.ctc_lo");

    auto tensors = model->encoder.get_all_tensors();
    for(auto &kv : model->ctc.get_all_tensors())
        tensors.insert(std::move(kv));

    ggml_model_weight_alloc(model->ctx, backend, model->buf_weights, tensors);
   
    return SUCCESS;
}

int load_sensevoice_model(struct ggml_handle_t *ggml_handle, const char *model_path, int model_size)
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

    struct sensevoice_state_t *state = new sensevoice_state_t; 
    if(NULL == state)
    {
        LOG_ERROR("sensevoice state error %s", strerror(errno));
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

    struct sensevoice_model_t *model = nullptr;
    if(model_size)
    {
        model = (struct sensevoice_model_t *)((struct ggml_handle_t *)model_path)->model;
        ggml_handle->is_dup = 1;
    }
    else
    {
        model = new sensevoice_model_t; 
        gguf_loader loader(model_path);
        load_sensevoice_tokenizer(loader, model->vocab);
        load_sensevoice_weight(loader, model, backend);
        ggml_handle->is_dup = 0;
    }

	//struct gguf_context *gguf_ctx = gguf_init_from_file(model_path, gguf_params);
    if(NULL == model)
    {
        LOG_ERROR("sensevoice malloc error %s", strerror(errno));
        delete params;
        return ERROR;
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

    ggml_handle->in_nodes = 1;
    set_shape(&ggml_handle->input_shape[0], TENSOR_FLOAT32, 1, -1);

    ggml_handle->out_nodes = 1;
    set_shape(&ggml_handle->output_shape[0], TENSOR_INT8, 1, params->n_max_audio_length);
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


        model->encoder.embed.compute(hparams->language_id, hparams->use_itn, position, embedding);
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


int sensevoice_inference(struct ggml_handle_t *ggml_handle, matrix_t **input_matrix, matrix_t **output_matrix, void *param)
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

    sensevoice_pcm2feature_with_state(state->feature, speech_segment, false, ggml_handle->n_thread); 

    sensevoice_encoder_inference(model, params, state, ggml_handle->n_thread);
    sensevoice_decoder_ctc_inference(model, state, ggml_handle->n_thread);

    sensevoice_print_output(state, model->vocab, false, output);

    if(output_matrix[0]->shape.size >= output.size())
    {
        memcpy(output_matrix[0]->data, output.c_str(), output.size());
        output_matrix[0]->shape.dims[0] = output.size();
    }

    return SUCCESS;
}

void unload_sensevoice_model(struct ggml_handle_t *ggml_handle)
{
    if(ggml_handle == NULL)
        return;

    struct sensevoice_params_t *params = (struct sensevoice_params_t *)ggml_handle->params;
    struct sensevoice_model_t *model = (struct sensevoice_model_t *)ggml_handle->model;
    struct sensevoice_state_t *state = (struct sensevoice_state_t *)ggml_handle->state;

    // Free state
    if(state)
    {
        // Free model weight buffer (must be freed before backends)
        if(state->model_buffer)
        {
            ggml_backend_buffer_free(state->model_buffer);
            state->model_buffer = nullptr;
        }
        if(state->model_ctx)
        {
            ggml_free(state->model_ctx);
            state->model_ctx = nullptr;
        }

        // Free KV cache
        if(state->kv_pad.ctx)
        {
            ggml_free(state->kv_pad.ctx);
            state->kv_pad.ctx = nullptr;
        }
        if(state->kv_pad.buffer)
        {
            ggml_backend_buffer_free(state->kv_pad.buffer);
            state->kv_pad.buffer = nullptr;
        }

        // Free feature context
        if(state->feature.ctx)
        {
            ggml_free(state->feature.ctx);
            state->feature.ctx = nullptr;
        }

        // Free schedulers
        if(state->sched_encode.sched)
        {
            ggml_backend_sched_free(state->sched_encode.sched);
            state->sched_encode.sched = nullptr;
        }
        if(state->sched_decode.sched)
        {
            ggml_backend_sched_free(state->sched_decode.sched);
            state->sched_decode.sched = nullptr;
        }

        // Note: backends are managed by ggml_handle, don't free them here
        state->backends.clear();

        delete state;
        ggml_handle->state = nullptr;
    }

    if(model && !ggml_handle->is_dup)
    {
        if(model->buf_weights)
        {
            ggml_backend_buffer_free(model->buf_weights);
        }

        if(model->ctx)
        {
            ggml_free(model->ctx);
        }

        delete model;
        ggml_handle->model = nullptr;
    }

    // Free params
    if(params)
    {
        delete params;
        ggml_handle->params = nullptr;
    }
}


