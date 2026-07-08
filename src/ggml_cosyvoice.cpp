#include "base.h"
#include "ggml_model.h"
#include "gguf_loader.h"
#include "ggml_cosyvoice.h"

#define COSYVOICE_DEFAULT_LLM_MAX_SEQ_LEN 4096

void get_cosyvoice_default_params(struct cosyvoice_params_t *params)
{
    params->flow_use_flash_attn = true;
    params->llm_use_flash_attn = true;

    // llm_kv_cache_type

    params->n_batch = 256;
    params->n_max_seq = COSYVOICE_DEFAULT_LLM_MAX_SEQ_LEN;

    //params->seed = ();

    // builtin_sampler_rng_policy
    // sampler
    // sampler_ctx
}


//prompt_file()


ggml_cgraph *cosyvoice_build_cgraph(struct cosyvoice_model_t *model, struct cosyvoice_state_t *state)
{
#if 0
    struct ggml_init_params params = {
        .mem_size   = state->build_buf_size,
        .mem_buffer = state->build_buf,
        .no_alloc   = true,
    };

    if(state->ctx_build)
        ggml_free(state->ctx_build);

    state->ctx_build = ggml_init(params);

    if(!state->ctx_build)
    {
        LOG_ERROR("ctx build ggml init error");
        return NULL;
    }

    ggml_tensor *input = ggml_new_tensor_1d(state->ctx_build, GGML_TYPE_F32, VAD_CHUNK_SIZE);
    // chunk size must be 576 before pad
    ggml_set_name(input, "audio_chunk");
    ggml_set_input(input);

    ggml_tensor *x = model->stft.build_cgraph(state->ctx_build, input);

    //to do encoder
    {
        x = ggml_conv_1d(state->ctx_build, model->encoders[0].weight, x, 1, 1, 1);
        x = ggml_add(state->ctx_build, x, ggml_cont(state->ctx_build, ggml_transpose(state->ctx_build, model->encoders[0].bias)));
        x = ggml_relu(state->ctx_build, x);

        x = ggml_conv_1d(state->ctx_build, model->encoders[1].weight, x, 2, 1, 1);
        x = ggml_add(state->ctx_build, x,  ggml_cont(state->ctx_build, ggml_transpose(state->ctx_build, model->encoders[1].bias)));
        x = ggml_relu(state->ctx_build, x);

        x = ggml_conv_1d(state->ctx_build, model->encoders[2].weight, x, 2, 1, 1);
        x = ggml_add(state->ctx_build, x,  ggml_cont(state->ctx_build, ggml_transpose(state->ctx_build, model->encoders[2].bias)));
        x = ggml_relu(state->ctx_build, x);

        x = ggml_conv_1d(state->ctx_build, model->encoders[3].weight, x, 1, 1, 1);
        x = ggml_add(state->ctx_build, x,  ggml_cont(state->ctx_build, ggml_transpose(state->ctx_build, model->encoders[3].bias)));
        x = ggml_relu(state->ctx_build, x);

    }

    x = model->rnn.build_cgraph(state->ctx_build, x);

    x = ggml_relu(state->ctx_build, x);
    x = ggml_conv_1d(state->ctx_build, model->decoder.weight, ggml_cont(state->ctx_build, ggml_transpose(state->ctx_build, x)), 1, 0, 1);
    x = ggml_add(state->ctx_build, x, ggml_transpose(state->ctx_build, model->decoder.bias));
    ggml_set_name(x, "decoder_out");
    ggml_tensor *output = ggml_sigmoid(state->ctx_build, x);
    ggml_set_name(output, "logit");

    ggml_cgraph *gf = ggml_new_graph(state->ctx_build);
    ggml_build_forward_expand(gf, output);

    return gf;
#endif
}

int load_cosyvoice_model(struct ggml_handle_t *ggml_handle, const char *model_data, int model_size)
{
#if 0
    struct cosyvoice_params_t *params = new cosyvoice_params_t;
    if(NULL == params)
    {
        LOG_ERROR("cosyvoice params error %s", strerror(errno));
        return ERROR;
    }
    get_cosyvoice_default_params(params);

    struct cosyvoice_state_t *state = new cosyvoice_state_t;
    if(NULL == state)
    {
        LOG_ERROR("cosyvoice state error %s", strerror(errno));
        delete params;
        return ERROR;
    }
    memset(state, 0, sizeof(struct cosyvoice_state_t));

    struct cosyvoice_model_t *model = nullptr;

    /* model */
    if(model_size > 0)
    {
        model = (struct cosyvoice_model_t *)((struct ggml_handle_t *)model_data)->model;
        ggml_handle->is_model_alloc = 0;
    }
    else
    {
        std::string prefix = "_model";
        model = new cosyvoice_model_t;
        if(NULL == model)
        {
            LOG_ERROR("cosyvoice malloc error %s", strerror(errno));
            delete params;
            return ERROR;
        }
        //memset(model, 0, sizeof(struct cosyvoice_model_t));
        LOG_DEBUG("model %p", model);

        LOG_DEBUG("model_data %s", model_data);
        gguf_loader loader(model_data);

        model->ctx = ggml_init(ggml_init_params{ .mem_size = ggml_graph_overhead() * GGML_DEFAULT_GRAPH_SIZE, .no_alloc = true });

        model->stft.onload(loader, prefix);

        //LOG_DEBUG("stft.forward_basis_buffer %p", model->stft.forward_basis_buffer);

        model->encoders.resize(params->n_encoder_layer);
        for(int i = 0; i < params->n_encoder_layer; ++i)
        {
            auto& layer = model->encoders[i];
            std::string name = prefix + ".encoder."  + std::to_string(i) + ".reparam_conv";
            layer.onload(loader, name);
        }
        model->rnn.onload(loader, prefix + ".decoder");
        model->decoder.onload(loader, prefix + ".decoder.decoder.2");

#if 0
        model->buf_weights = ggml_backend_alloc_ctx_tensors(model->ctx, ggml_handle->backend);
        if(!model->buf_weights)
        {
            LOG_ERROR("buf weights ggml_backend_alloc_ctx_tensors error");
            return ERROR;
        }
#endif
        auto tensors = model->stft.get_all_tensors();

        for(auto &layer : model->encoders)
            for(auto &kv : layer.get_all_tensors())
                tensors.insert(std::move(kv));

        for(auto &kv : model->rnn.get_all_tensors())
            tensors.insert(std::move(kv));

        for(auto &kv : model->decoder.get_all_tensors())
            tensors.insert(std::move(kv));

        ggml_model_weight_alloc(model->ctx, ggml_handle->backend, model->buf_weights, tensors);
        ggml_handle->is_model_alloc = 1;
    }

    /* cache */
    struct ggml_init_params cache_params = {
        .mem_size = sizeof(float) * params->lstm_state_dim * 2 + 256,
        .no_alloc = true,
    };
    state->ctx_cache = ggml_init(cache_params);
    if(!state->ctx_cache)
    {
        LOG_ERROR("ctx cache ggml init error");
        return ERROR;
    }

    state->vad_lstm_context = ggml_new_tensor_1d(state->ctx_cache, GGML_TYPE_F32, params->lstm_state_dim);
    state->vad_lstm_hidden_state = ggml_new_tensor_1d(state->ctx_cache, GGML_TYPE_F32, params->lstm_state_dim);
    state->buf_cache = ggml_backend_alloc_ctx_tensors(state->ctx_cache, ggml_handle->backend); //cpu
    if (!state->buf_cache) 
    {
        LOG_ERROR("state buf cache alloc error");
        return ERROR;
    }

    float zeros[params->lstm_state_dim] = {0};
    ggml_backend_tensor_set(state->vad_lstm_context, zeros, 0, sizeof(zeros));
    ggml_backend_tensor_set(state->vad_lstm_hidden_state, zeros, 0, sizeof(zeros));

    state->build_buf_size = ggml_graph_overhead() * GGML_DEFAULT_GRAPH_SIZE;
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
        GGML_DEFAULT_GRAPH_SIZE,
        true,
        true
    );  

    ggml_cgraph *gf = cosyvoice_build_cgraph(model, state);
    if (!ggml_backend_sched_alloc_graph(state->sched, gf)) 
    {
        LOG_ERROR("sched_alloc_graph error");
        return ERROR;
    }   

    //set_graph_backend(gf, state->sched, ggml_handle->backend, nullptr, -1); //to do 

    //to do cpu ?
    ggml_backend_sched_reset(state->sched);

    ggml_handle->params = (void *)params;
    ggml_handle->model = (void *)model;
    ggml_handle->state = (void *)state;

    ggml_handle->in_nodes = 2;
    ggml_handle->input_names[0] = strdup("audio_chunk");
    set_shape(&ggml_handle->input_shape[0], TENSOR_FLOAT32, 1, 640);
    ggml_handle->input_names[1] = strdup("audio_cache");
    set_shape(&ggml_handle->input_shape[1], TENSOR_FLOAT32, 1, 512);

    ggml_handle->output_names[0] = strdup("logit");
    ggml_handle->out_nodes = 1;
    //to do no use
    set_shape(&ggml_handle->output_shape[0], TENSOR_FLOAT32, 1, params->max_speech_duration_ms * 16);
#endif
    return SUCCESS;
}

int cosyvoice_frontend_process(struct ggml_handle_t *ggml_handle, matrix_t **input_matrix)
{
    return SUCCESS;
}

int cosyvoice_backend_process(struct ggml_handle_t *ggml_handle, float speech_prob)
{
    return SUCCESS;
}


int cosyvoice_inference(struct ggml_handle_t *ggml_handle, matrix_t **input_matrix,
                        matrix_t **output_matrix)
{

    struct cosyvoice_params_t *params = (struct cosyvoice_params_t *)ggml_handle->params;
    struct cosyvoice_model_t *model = (struct cosyvoice_model_t *)ggml_handle->model;
    struct cosyvoice_state_t *state = (struct cosyvoice_state_t *)ggml_handle->state;

    ggml_cgraph *gf = cosyvoice_build_cgraph(model, state);

    if (!ggml_backend_sched_alloc_graph(state->sched, gf)) 
    {
        LOG_ERROR("ggml_backend_sched_alloc_graph error");
        return ERROR;
    }

	cosyvoice_frontend_process(ggml_handle, input_matrix);

#if 0
    // set the input
    struct ggml_tensor *data = ggml_graph_get_tensor(gf, "audio_chunk");
    ggml_backend_tensor_set(data, input_matrix[0]->data, 0, ggml_nbytes(data));

    //to do 
    struct ggml_tensor *in_lstm_context = ggml_graph_get_tensor(gf, "in_lstm_context");
    struct ggml_tensor *in_lstm_hidden_state = ggml_graph_get_tensor(gf, "in_lstm_hidden_state");
    ggml_backend_tensor_copy(state->vad_lstm_context, in_lstm_context);
    ggml_backend_tensor_copy(state->vad_lstm_hidden_state, in_lstm_hidden_state);
#endif

#if 0
    if (!ggml_graph_compute_helper(sched, gf, ggml_handle->n_thread))
    {
        LOG_ERROR("cosyvoice_inference ggml_graph_compute_helper error");
        return ERROR;
    }
#endif
    ggml_backend_sched_graph_compute(state->sched, gf);

#if 0
    // save output state
    struct ggml_tensor *lstm_context = ggml_graph_get_tensor(gf, "out_lstm_context");
    ggml_backend_tensor_copy(lstm_context, state->vad_lstm_context);
    struct ggml_tensor *lstm_hidden_state = ggml_graph_get_tensor(gf, "out_lstm_hidden_state");
    ggml_backend_tensor_copy(lstm_hidden_state, state->vad_lstm_hidden_state);

    float speech_prob = 0.0;

    ggml_backend_tensor_get(ggml_graph_get_tensor(gf, "logit"), &speech_prob, 0, sizeof(speech_prob));

    cosyvoice_backend_process(ggml_handle, speech_prob);
    ggml_backend_sched_reset(state->sched);
#endif
    return SUCCESS;
}


void unload_cosyvoice_model(struct ggml_handle_t *ggml_handle)
{
    struct cosyvoice_params_t *params = (struct cosyvoice_params_t *)ggml_handle->params;
    struct cosyvoice_model_t *model = (struct cosyvoice_model_t *)ggml_handle->model;
    struct cosyvoice_state_t *state = (struct cosyvoice_state_t *)ggml_handle->state;

    if (state->sched) 
    {
        ggml_backend_sched_free(state->sched);
    }   

    if (state->ctx_build) 
    {
        ggml_free(state->ctx_build);
    }   

    if (state->build_buf) 
    {
        free(state->build_buf);
    }   

    if (state->buf_cache) 
    {
        ggml_backend_buffer_free(state->buf_cache);
    }   

    if (state->ctx_cache) 
    {
        ggml_free(state->ctx_cache);
    }   

    memset(state, 0, sizeof(struct cosyvoice_state_t));

    if(model->buf_weights)
    {
        ggml_backend_buffer_free(model->buf_weights);
    }

    if(model->ctx)
    {
        ggml_free(model->ctx);
    }
    memset(model, 0, sizeof(struct cosyvoice_model_t));

    delete state;
    delete model;
    delete params;

    ggml_handle->state = NULL;
    ggml_handle->model = NULL;
    ggml_handle->params = NULL;
}

