#include "base.h"
#include "ggml_model.h"
#include "gguf_loader.h"
#include "ggml_sensevoice.h"

#define SENSEVOICE_FEATURES_DIM 560
#define SENSEVOICE_CHUNK_SIZE 20  //to do 

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
}

//static bool 
//init_sensevoice_kv_cache(); 

ggml_cgraph *sensevoice_build_cgraph(struct sensevoice_model_t *model, struct sensevoice_state_t *state, struct sensevoice_params_t *hparams)
{
    struct ggml_init_params params = {
        .mem_size   = state->build_buf_size,
        .mem_buffer = state->build_buf,
        .no_alloc   = true,
    };
    LOG_DEBUG("state->build_buf_size %d", state->build_buf_size);

    int flash_attn = 0;

    if(state->ctx_build)
        ggml_free(state->ctx_build);

    state->ctx_build = ggml_init(params);

    if(!state->ctx_build)
    {
        LOG_ERROR("ctx build ggml init error");
        return NULL;
    }

    ggml_tensor *input = state->feature;
    ggml_set_name(input, "feature");
    ggml_set_input(input);

    LOG_DEBUG("hparams->fsmn_kernel_size %d flash_attn = %d", hparams->fsmn_kernel_size, flash_attn);
    //hparam to do 
    ggml_tensor *x = model->encoder.build_cgraph(state->ctx_build, input, hparams->fsmn_kernel_size, flash_attn);

    auto [probs, argmax_logit]  = model->ctc.build_cgraph(state->ctx_build, x);

    ggml_set_name(probs, "probs");
    ggml_set_output(probs);

    ggml_set_name(argmax_logit, "argmax_logit");
    ggml_set_output(argmax_logit);

    ggml_cgraph *gf = ggml_new_graph_custom(state->ctx_build, 8192, false);
    //ggml_cgraph *gf = ggml_new_graph_custom(state->ctx_build, 8192);
    ggml_build_forward_expand(gf, argmax_logit);
    return gf;
}

void unload_sensevoice_model(struct ggml_handle_t *ggml_handle)
{
    struct sensevoice_params_t *params = (struct sensevoice_params_t *)ggml_handle->params;
    struct sensevoice_model_t *model = (struct sensevoice_model_t *)ggml_handle->model;
    struct sensevoice_state_t *state = (struct sensevoice_state_t *)ggml_handle->state;

    if (state->sched)
    {
        ggml_backend_sched_free(state->sched);
    }

    if (state->ctx_build)
    {
        ggml_free(state->ctx_build);
    }

    if (state->buf_cache)
    {
        ggml_backend_buffer_free(state->buf_cache);
    }

    if (state->ctx_cache)
    {
        ggml_free(state->ctx_cache);
    }

    if (state->build_buf)
    {
        free(state->build_buf);
    }

    memset(state, 0, sizeof(struct sensevoice_state_t));

    if(ggml_handle->is_model_alloc) //to do 
    {
        if(model->buf_weights)
        {
            ggml_backend_buffer_free(model->buf_weights);
        }

        if(model->ctx)
        {
            ggml_free(model->ctx);
        }
    }

    memset(model, 0, sizeof(struct sensevoice_model_t));

    delete state;
    delete model;
    delete params;

    ggml_handle->state = NULL;
    ggml_handle->model = NULL;
    ggml_handle->params = NULL;
}

int load_sensevoice_tokenizer()
{
#if 0
    GGML_ASSERT(loader.get_string("tokenizer.model.type") == "BPE");

    const auto token_idx = gguf_find_key(loader, "tokenizer.vocab.tokens");

    const int* toktypes = nullptr;
    const auto toktype_idx = gguf_find_key(loader, "tokenizer.vocab.token_types");
    toktypes = (const int*)gguf_get_arr_data(loader, toktype_idx);

    auto n_tokens = gguf_get_arr_n(loader, token_idx);
    pimpl_->id_to_token.resize(n_tokens);

    for (int i = 0; i != n_tokens; i++) {
        std::string word = gguf_get_arr_str(loader, token_idx, i);
        if (word.empty())
            word = "[EMPTY_" + std::to_string(i) + "]";

        pimpl_->token_to_id[word] = i;

        auto& token_data = pimpl_->id_to_token[i];
        token_data.text = std::move(word);
        token_data.type = static_cast<token_type>(toktypes[i]);

        if (token_data.type == TOKEN_TYPE_CONTROL)
            pimpl_->cache_special_tokens.push_back(i);
    }   
    GGML_ASSERT(pimpl_->id_to_token.size() == pimpl_->token_to_id.size());

    const auto merges_keyidx = gguf_find_key(loader, "tokenizer.model.merges");
    const auto n_merges = gguf_get_arr_n(loader, merges_keyidx);
    for (int i = 0; i < n_merges; i++) {
        const std::string_view word = gguf_get_arr_str(loader, merges_keyidx, i); 

        std::string_view first;
        std::string_view second;

        const size_t pos = word.find(' ', 1); 

        if (pos != std::string_view::npos) {
            first = word.substr(0, pos);
            second = word.substr(pos + 1);
        }

        pimpl_->bpe_ranks.emplace(std::make_pair(first, second), i);
    }

    pimpl_->tokenizer = std::make_unique<llm_tokenizer_bpe>(loader);

    std::sort(pimpl_->cache_special_tokens.begin(), pimpl_->cache_special_tokens.end(),
        [&](const int a, const int b) {
            return pimpl_->id_to_token[a].text.size() > pimpl_->id_to_token[b].text.size();
        }
    );
#endif
}

int load_sensevoice_model(struct ggml_handle_t *ggml_handle, const char *model_data, int model_size)
{
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
    memset(state, 0, sizeof(struct sensevoice_state_t));

    struct sensevoice_model_t *model = nullptr;

    /* model */
    if(model_size > 0)
    {
        model = (struct sensevoice_model_t *)((struct ggml_handle_t *)model_data)->model;
        ggml_handle->is_model_alloc = 0;
    }
    else
    {
        std::string prefix = "encoder";
        model = new sensevoice_model_t;
        if(NULL == model)
        {
            LOG_ERROR("sensevoice malloc error %s", strerror(errno));
            delete params;
            return ERROR;
        }

        gguf_loader loader(model_data);
        model->ctx = ggml_init(ggml_init_params{ .mem_size = ggml_graph_overhead() * GGML_DEFAULT_GRAPH_SIZE, .no_alloc = true });

        model->encoder.onload(loader, prefix);
        model->ctc.onload(loader, "ctc.ctc_lo");

        auto tensors = model->encoder.get_all_tensors();
        for(auto &kv : model->ctc.get_all_tensors())
            tensors.insert(std::move(kv));

        ggml_model_weight_alloc(model->ctx, ggml_handle->backend, model->buf_weights, tensors);
        ggml_handle->is_model_alloc = 1;
    }

    /* vocab */
    // tokenizer.ggml.tokens

    /* cache */
    struct ggml_init_params cache_params = {
        .mem_size = sizeof(float) * SENSEVOICE_CHUNK_SIZE * SENSEVOICE_FEATURES_DIM + 256,
        .no_alloc = true,
    };
    //to do dynamic
    state->ctx_cache = ggml_init(cache_params);
    if(!state->ctx_cache)
    {
        LOG_ERROR("ctx cache ggml init error");
        return ERROR;
    }

    state->feature = ggml_new_tensor_2d(state->ctx_cache, GGML_TYPE_F32, SENSEVOICE_FEATURES_DIM, SENSEVOICE_CHUNK_SIZE);
    state->buf_cache = ggml_backend_alloc_ctx_tensors(state->ctx_cache, ggml_handle->backend); //cpu
    if (!state->buf_cache)
    {
        LOG_ERROR("state buf cache alloc error");
        return ERROR;
    }

    float zeros[SENSEVOICE_FEATURES_DIM * SENSEVOICE_CHUNK_SIZE] = {0};
    ggml_backend_tensor_set(state->feature, zeros, 0, sizeof(zeros));

    /* kqv */
    state->build_buf_size = ggml_graph_overhead() * GGML_DEFAULT_GRAPH_SIZE * 2;
    state->build_buf = malloc(state->build_buf_size);
    if(!state->build_buf)
    {
        LOG_ERROR("build buf malloc size %ld error %d", state->build_buf_size, strerror(errno));
        return ERROR;
    }

    /* state */
    ggml_backend_t backends[] = { ggml_handle->backend, ggml_handle->cpu_backend };
    state->sched = ggml_backend_sched_new(
        backends,
        nullptr,
        ggml_handle->backend == ggml_handle->cpu_backend ? 1 : 2,
        8192, 
        //GGML_DEFAULT_GRAPH_SIZE,
        true,
        true
    );

    ggml_backend_sched_reset(state->sched);

    ggml_cgraph *gf = sensevoice_build_cgraph(model, state, params);

    LOG_DEBUG("state->sched %p gf %p ", state->sched, gf);
    if (!ggml_backend_sched_alloc_graph(state->sched, gf))
    {
        LOG_ERROR("sched_alloc_graph error");
        return ERROR;
    }

    //set_graph_backend(gf, state->sched, ggml_handle->backend, nullptr, -1); //to do 

    //to do cpu ?

    ggml_handle->in_nodes = 4;
    ggml_handle->input_names[0] = strdup("speech");
    set_shape(&ggml_handle->input_shape[0], TENSOR_FLOAT32, 3,  -1, -1, 560);

    ggml_handle->input_names[1] = strdup("speech_lengths");
    set_shape(&ggml_handle->input_shape[1], TENSOR_INT32, 1, -1);

    ggml_handle->input_names[2] = strdup("language");
    set_shape(&ggml_handle->input_shape[2], TENSOR_INT32, 1, -1);

    ggml_handle->input_names[3] = strdup("textnorm");
    set_shape(&ggml_handle->input_shape[3], TENSOR_INT32, 1, -1);

    ggml_handle->out_nodes = 2;
    ggml_handle->output_names[0] = strdup("ctc_logits");
    set_shape(&ggml_handle->output_shape[0], TENSOR_FLOAT32, 3, -1, -1, 25055);
    ggml_handle->output_names[1] = strdup("encoder_out_lens");
    set_shape(&ggml_handle->output_shape[1], TENSOR_FLOAT32, 1, -1);

    ggml_handle->params = (void *)params;
    ggml_handle->model = (void *)model;
    ggml_handle->state = (void *)state;

    return SUCCESS;
}

int sensevoice_frontend_process(struct ggml_handle_t *ggml_handle, matrix_t **input_matrix)
{
    return SUCCESS;
}

int sensevoice_backend_process(struct ggml_handle_t *ggml_handle, float speech_prob)
{
    return SUCCESS;
}

int sensevoice_inference(struct ggml_handle_t *ggml_handle, matrix_t **input_matrix,
                        matrix_t **output_matrix)
{

    struct sensevoice_params_t *params = (struct sensevoice_params_t *)ggml_handle->params;
    struct sensevoice_model_t *model = (struct sensevoice_model_t *)ggml_handle->model;
    struct sensevoice_state_t *state = (struct sensevoice_state_t *)ggml_handle->state;

    ggml_cgraph *gf = sensevoice_build_cgraph(model, state, params);
    LOG_DEBUG("sched->galloc = %p", (void*)state->sched->galloc);

    LOG_DEBUG("state->sched %p gf %p ", state->sched, gf);
    if (!ggml_backend_sched_alloc_graph(state->sched, gf))
    {
        LOG_ERROR("ggml_backend_sched_alloc_graph error");
        return ERROR;
    }


    sensevoice_frontend_process(ggml_handle, input_matrix);


#if 0
    // set the input
    struct ggml_tensor *data = ggml_graph_get_tensor(gf, "audio_chunk");
    ggml_backend_tensor_set(data, input_matrix[0]->data, 0, ggml_nbytes(data));

    //to do 
    struct ggml_tensor *in_lstm_context = ggml_graph_get_tensor(gf, "in_lstm_context");
    struct ggml_tensor *in_lstm_hidden_state = ggml_graph_get_tensor(gf, "in_lstm_hidden_state");
    ggml_backend_tensor_copy(state->vad_lstm_context, in_lstm_context);
    ggml_backend_tensor_copy(state->vad_lstm_hidden_state, in_lstm_hidden_state);

#if 0
    if (!ggml_graph_compute_helper(sched, gf, ggml_handle->n_thread))
    {
        LOG_ERROR("sensevoice_inference ggml_graph_compute_helper error");
        return ERROR;
    }
#endif
    ggml_backend_sched_graph_compute(state->sched, gf);

    // save output state
    struct ggml_tensor *lstm_context = ggml_graph_get_tensor(gf, "out_lstm_context");
    ggml_backend_tensor_copy(lstm_context, state->vad_lstm_context);
    struct ggml_tensor *lstm_hidden_state = ggml_graph_get_tensor(gf, "out_lstm_hidden_state");
    ggml_backend_tensor_copy(lstm_hidden_state, state->vad_lstm_hidden_state);

    float speech_prob = 0.0;

    ggml_backend_tensor_get(ggml_graph_get_tensor(gf, "logit"), &speech_prob, 0, sizeof(speech_prob));

    sensevoice_backend_process(ggml_handle, speech_prob);
    ggml_backend_sched_reset(state->sched);
#endif
    return SUCCESS;
}

