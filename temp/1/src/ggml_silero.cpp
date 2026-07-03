#include "base.h"
#include "ggml_model.h"
#include "gguf_loader.h"
#include "ggml_silero.h"

void get_silero_default_params(struct silero_params_t *params)
{
    params->n_encoder_layer = 4;
    params->lstm_state_dim = 128;

    params->n_batch = 1;

    params->threshold      = 0.5f;
    params->neg_threshold = 0.35f;
    params->min_speech_duration_ms = 250;
    params->max_speech_duration_ms = 15000;
    params->min_silence_duration_ms = 100;
    params->speech_pad_ms = 30; 
}

ggml_cgraph *silero_build_cgraph(struct silero_model_t *model, struct silero_state_t *state)
{
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

    //ggml_tensor *x = model->stft.build_cgraph(state->ctx_build, input);

    ggml_tensor *cur;
    ggml_context *ctx0 = state->ctx_build;

    LOG_DEBUG("model %p", model);
    LOG_DEBUG("model->stft.forward_basis_buffer %p", model->stft.forward_basis_buffer);
    // stft
    {   

        cur = ggml_conv_1d(ctx0, model->stft.forward_basis_buffer, input, 128, 0, 1); 
        // chunk operation by ggml view, equals torch.chunk(x, 2) in pytorch
        struct ggml_tensor * real_part = ggml_view_2d(ctx0, cur, cur->ne[0], cur->ne[1] / 2, cur->nb[1], 0);
        ggml_set_name(real_part, "real_part");
        struct ggml_tensor * image_part = ggml_view_2d(ctx0, cur, cur->ne[0], cur->ne[1] / 2, cur->nb[1], cur->nb[0] * cur->ne[0] * cur->ne[1] / 2);
        ggml_set_name(image_part, "image_part");
        // magnitude, equals torch.sqrt(real_part ** 2 + imag_part ** 2)
        cur = ggml_sqrt(ctx0,
                        ggml_add(ctx0,
                                 ggml_mul(ctx0, real_part, real_part),
                                 ggml_mul(ctx0, image_part, image_part)
                                 )
                        );
        ggml_set_name(cur, "magnitude");

    }   


    //to do encoder
    {
#if 0
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
#endif
            cur = ggml_conv_1d(ctx0, model->encoders[0].weight, cur, 1, 1, 1);
            cur = ggml_add(ctx0, cur, ggml_cont(ctx0, ggml_transpose(ctx0, model->encoders[0].bias)));
            cur = ggml_relu(ctx0, cur);

            cur = ggml_conv_1d(ctx0, model->encoders[1].weight, cur, 2, 1, 1);
            cur = ggml_add(ctx0, cur,  ggml_cont(ctx0, ggml_transpose(ctx0, model->encoders[1].bias)));
            cur = ggml_relu(ctx0, cur);

            cur = ggml_conv_1d(ctx0, model->encoders[2].weight, cur, 2, 1, 1);
            cur = ggml_add(ctx0, cur,  ggml_cont(ctx0, ggml_transpose(ctx0, model->encoders[2].bias)));
            cur = ggml_relu(ctx0, cur);

            cur = ggml_conv_1d(ctx0, model->encoders[3].weight, cur, 1, 1, 1);
            cur = ggml_add(ctx0, cur,  ggml_cont(ctx0, ggml_transpose(ctx0, model->encoders[3].bias)));
            cur = ggml_relu(ctx0, cur);

    }

    //decoder
    {
        struct ggml_tensor* in_lstm_hidden_state = ggml_new_tensor_1d(ctx0, cur->type, cur->ne[1]);
        struct ggml_tensor*  in_lstm_context = ggml_new_tensor_1d(ctx0, cur->type, cur->ne[1]);

        struct ggml_tensor* out_lstm_hidden_state;
        struct ggml_tensor*  out_lstm_context;

        ggml_set_name(in_lstm_context, "in_lstm_context");
        ggml_set_name(in_lstm_hidden_state, "in_lstm_hidden_state");

        // lstm cell
        // ref: https://github.com/pytorch/pytorch/blob/1a93b96815b5c87c92e060a6dca51be93d712d09/aten/src/ATen/native/RNN.cpp#L298-L304
        // gates = x @ self.weight_ih.T + self.bias_ih + hx[0] @ self.weight_hh.T + self.bias_hh
        // chunked_gates = gates.chunk(4, dim=-1)
        // ingate = torch.sigmoid(chunked_gates[0])
        // forgetgate = torch.sigmoid(chunked_gates[1])
        // cellgate = torch.tanh(chunked_gates[2])
        // outgate = torch.sigmoid(chunked_gates[3])
        // cy = forgetgate * hx[1] + ingate * cellgate
        // hy = outgate * torch.tanh(cy)

        //cur = ggml_cont(ctx0, cur);
        cur = ggml_cont(ctx0, ggml_transpose(ctx0, cur));

        struct ggml_tensor *gates = ggml_add(
                ctx0,
                ggml_add(ctx0, ggml_mul_mat(ctx0,
                                            model->rnn.lstm_weight_ih,
                                            cur),
                         model->rnn.lstm_bias_ih),

                ggml_add(ctx0, ggml_mul_mat(ctx0,
                                            model->rnn.lstm_weight_hh,
                                            in_lstm_hidden_state),
                         model->rnn.lstm_bias_hh));
        ggml_set_name(gates, "gates");

        struct ggml_tensor * input_gates = ggml_sigmoid(ctx0, ggml_view_2d(ctx0, gates, gates->ne[0] / 4, gates->ne[1] , gates->nb[1], 0));
        struct ggml_tensor * forget_gates = ggml_sigmoid(ctx0, ggml_view_2d(ctx0, gates, gates->ne[0] / 4, gates->ne[1], gates->nb[1], gates->nb[0] / 4 * gates->ne[0]));
        struct ggml_tensor * cell_gate = ggml_tanh(ctx0, ggml_view_2d(ctx0, gates, gates->ne[0] / 4, gates->ne[1], gates->nb[1], 2 * gates->nb[0] / 4 * gates->ne[0]));
        struct ggml_tensor * out_gates = ggml_sigmoid(ctx0, ggml_view_2d(ctx0, gates, gates->ne[0] / 4, gates->ne[1], gates->nb[1], 3 * gates->nb[0] / 4 * gates->ne[0]));

        ggml_set_name(input_gates, "input_gates");
        ggml_set_name(forget_gates, "forget_gates");
        ggml_set_name(cell_gate, "cell_gates");
        ggml_set_name(out_gates, "out_gates");

        out_lstm_context = ggml_add(ctx0,
                                          ggml_mul(ctx0, forget_gates, in_lstm_context),
                                          ggml_mul(ctx0, input_gates, cell_gate)
                                          );
        ggml_set_name(out_lstm_context, "out_lstm_context");
        ggml_set_output(out_lstm_context);
        out_lstm_hidden_state = ggml_mul(ctx0, out_gates, ggml_tanh(ctx0, out_lstm_context));
        ggml_set_name(out_lstm_hidden_state, "out_lstm_hidden_state");
        ggml_set_output(out_lstm_hidden_state);

        cur = ggml_relu(ctx0, out_lstm_hidden_state);
        cur = ggml_conv_1d(ctx0, model->decoder.weight, ggml_cont(ctx0, ggml_transpose(ctx0, cur)), 1, 0, 1);
        cur = ggml_add(ctx0, cur, ggml_transpose(ctx0, model->decoder.bias));
        ggml_set_name(cur, "decoder_out");
        cur = ggml_sigmoid(ctx0, cur);
        ggml_set_name(cur, "logit");

    }

    //x = model->rnn.build_cgraph(state->ctx_build, x);

    //x = ggml_relu(state->ctx_build, x);
    //x = ggml_conv_1d(state->ctx_build, model->decoder.weight, ggml_cont(state->ctx_build, ggml_transpose(state->ctx_build, x)), 1, 0, 1);
    //x = ggml_add(state->ctx_build, x, ggml_transpose(state->ctx_build, model->decoder.bias));
    //ggml_set_name(x, "decoder_out");
    //ggml_tensor *output = ggml_sigmoid(state->ctx_build, x);
    //ggml_set_name(output, "logit");

    ggml_cgraph *gf = ggml_new_graph(state->ctx_build);
    ggml_build_forward_expand(gf, cur);

    return gf;
}

int load_silero_model(struct ggml_handle_t *ggml_handle, const char *model_data, int model_size)
{
    struct silero_params_t *params = new silero_params_t;
    if(NULL == params)
    {
        LOG_ERROR("silero params error %s", strerror(errno));
        return ERROR;
    }
    get_silero_default_params(params);

    struct silero_state_t *state = new silero_state_t;
    if(NULL == state)
    {
        LOG_ERROR("silero state error %s", strerror(errno));
        delete params;
        return ERROR;
    }
    memset(state, 0, sizeof(struct silero_state_t));

    struct silero_model_t *model = nullptr;

    /* model */
    if(model_size > 0)
    {
        model = (struct silero_model_t *)((struct ggml_handle_t *)model_data)->model;
        ggml_handle->is_model_alloc = 0;
    }
    else
    {
        std::string prefix = "_model";
        struct silero_model_t *model = new silero_model_t;
        if(NULL == model)
        {
            LOG_ERROR("silero malloc error %s", strerror(errno));
            delete params;
            return ERROR;
        }
        memset(model, 0, sizeof(struct silero_model_t));
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


    ggml_cgraph *gf = silero_build_cgraph(model, state);
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
    return SUCCESS;
}

int silero_inference(struct ggml_handle_t *ggml_handle, matrix_t **input_matrix,
                        matrix_t **output_matrix)
{

    struct silero_params_t *params = (struct silero_params_t *)ggml_handle->params;
    struct silero_model_t *model = (struct silero_model_t *)ggml_handle->model;
    struct silero_state_t *state = (struct silero_state_t *)ggml_handle->state;

    ggml_cgraph *gf = silero_build_cgraph(model, state);

    if (!ggml_backend_sched_alloc_graph(state->sched, gf)) 
    {
        LOG_ERROR("ggml_backend_sched_alloc_graph error");
        return ERROR;
    }

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
        LOG_ERROR("silero_inference ggml_graph_compute_helper error");
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

    //silero_backend_process(ggml_handle, speech_prob);
    ggml_backend_sched_reset(state->sched);
    return SUCCESS;
}


void unload_silero_model(struct ggml_handle_t *ggml_handle)
{
    struct silero_params_t *params = (struct silero_params_t *)ggml_handle->params;
    struct silero_model_t *model = (struct silero_model_t *)ggml_handle->model;
    struct silero_state_t *state = (struct silero_state_t *)ggml_handle->state;

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

    memset(state, 0, sizeof(struct silero_state_t));

    if(model->buf_weights)
    {
        ggml_backend_buffer_free(model->buf_weights);
    }

    if(model->ctx)
    {
        ggml_free(model->ctx);
    }
    memset(model, 0, sizeof(struct silero_model_t));

    delete state;
    delete model;
    delete params;

    ggml_handle->state = NULL;
    ggml_handle->model = NULL;
    ggml_handle->params = NULL;

}



