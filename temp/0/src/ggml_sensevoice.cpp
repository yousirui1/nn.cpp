#include "base.h"
#include "gguf_loader.hpp"
#include "ggml_sensevoice.hpp"
#include "ggml_handle.hpp"

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

ggml_cgraph *sensevoice_build_cgraph(struct sensevoice_model_t *model, struct sensevoice_state_t *state)
{
#if 0
    struct ggml_init_params params = {
            /*.mem_size   =*/state->meta.size(),
            /*.mem_buffer =*/state->meta.data(),
            /*.no_alloc   =*/true,
    };

    struct ggml_context *ctx0 = ggml_init(params);

    ggml_cgraph *gf = ggml_new_graph_custom(ctx0, SILERO_MAX_NODES, false);

    ggml_tensor *chunk = ggml_new_tensor_1d(ctx0, GGML_TYPE_F32, VAD_CHUNK_SIZE);
    // chunk size must be 576 before pad
    ggml_set_name(chunk, "audio_chunk");
    ggml_set_input(chunk);

    ggml_tensor *cur;
    // stft
    {
        cur = ggml_conv_1d(ctx0, model->stft.forward_basis_buffer, chunk, 128, 0, 1);
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

    // encoder
    {
        {
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

    ggml_set_output(cur);
    ggml_build_forward_expand(gf, cur);
    ggml_free(ctx0);
    return gf;
#endif
}

void unload_sensevoice_model(struct ggml_handle_t *ggml_handle)
{

}

int load_sensevoice_model(struct ggml_handle_t *ggml_handle, const char *model_path)
{
#if 0
    struct ggml_context *ctx;
    ggml_backend_buffer_t buffer = nullptr;

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

    gguf_loader loader(model_path);

    ctx = ggml_init(ggml_init_params{ .mem_size = ggml_graph_overhead() * GGML_DEFAULT_GRAPH_SIZE, .no_alloc = true });

    std::string prefix = "_model";

    model->stft.onload(loader, prefix);

    model->encoders.resize(params->n_encoder_layer);
    for(int i = 0; i < params->n_encoder_layer; ++i)
    {
        auto& layer = model->encoders[i];
        std::string name = prefix + ".encoder."  + std::to_string(i) + ".reparam_conv";
        layer.onload(loader, name);
    }
    model->rnn.onload(loader, prefix + ".decoder");
    model->decoder.onload(loader, prefix + ".decoder.decoder.2");

    auto tensors = model->stft.get_all_tensors();

    for(auto &layer : model->encoders)
        for(auto &kv : layer.get_all_tensors())
            tensors.insert(std::move(kv));

    for(auto &kv : model->rnn.get_all_tensors())
        tensors.insert(std::move(kv));

    for(auto &kv : model->decoder.get_all_tensors())
        tensors.insert(std::move(kv));

    auto buft = ggml_backend_get_default_buffer_type(ggml_handle->backend);
    auto alignment = ggml_backend_buft_get_alignment(buft);
    size_t mem_size = get_aligned_size(sizeof(float) * params->lstm_state_dim * 2, alignment);
    //LOG_DEBUG("mem_size %lld ", mem_size);

    for(const auto &[name, tensor] : tensors)
    {
        mem_size += get_aligned_size(ggml_backend_buft_get_alloc_size(buft, *tensor), alignment);
    }

    buffer = ggml_backend_buft_alloc_buffer(buft, mem_size);
    auto buffer_base = reinterpret_cast<char*>(ggml_backend_buffer_get_base(buffer));

    ggml_backend_buffer_set_usage(buffer, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);

    auto set_tensor = [&](ggml_tensor* tensor, const void* data, size_t size)
    {
        ggml_backend_tensor_set_async(ggml_handle->backend, tensor, data, 0, size);
        buffer_base += get_aligned_size(ggml_backend_buft_get_alloc_size(buft, tensor), alignment);
    };

    for (const auto& [name, tensor] : tensors)
    {
        size_t tensor_size = ggml_nbytes(*tensor);
        {
            auto new_tensor = ggml_new_tensor(ctx, (*tensor)->type, GGML_MAX_DIMS, (*tensor)->ne);
            ggml_backend_tensor_alloc(buffer, new_tensor, buffer_base);
            set_tensor(new_tensor, ggml_get_data(*tensor), tensor_size);
            *tensor = new_tensor;
        }
        ggml_set_param(*tensor);
        ggml_set_name(*tensor, name.c_str());
    }

    state->meta.resize(ggml_tensor_overhead() * SILERO_MAX_NODES + ggml_graph_overhead());

    ggml_backend_t backends[] = { ggml_handle->backend, ggml_handle->cpu_backend };
    state->sched = ggml_backend_sched_new(
        backends,
        nullptr,
        ggml_handle->backend == ggml_handle->cpu_backend ? 1 : 2,
        GGML_DEFAULT_GRAPH_SIZE,
        true,
        true
    );

    ggml_cgraph *gf = sensevoice_build_cgraph(model, state);
    ggml_backend_sched_alloc_graph(state->sched, gf);
    set_graph_backend(gf, state->sched, ggml_handle->backend, nullptr, -1);
    ggml_backend_sched_reset(state->sched);

    state->vad_lstm_context = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, params->lstm_state_dim);
    ggml_backend_tensor_alloc(buffer, state->vad_lstm_context, buffer_base);
    buffer_base += get_aligned_size(state->vad_lstm_context->nb[1], alignment);

    state->vad_lstm_hidden_state = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, params->lstm_state_dim);
    ggml_backend_tensor_alloc(buffer, state->vad_lstm_hidden_state, buffer_base);
    buffer_base += get_aligned_size(state->vad_lstm_hidden_state->nb[1], alignment);

    ggml_backend_synchronize(ggml_handle->backend);

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

int sensevoice_frontend_process(struct ggml_handle_t *ggml_handle, matrix_t **input_matrix)
{
#if 0
    int chunk_size = 512;
    int context_size = 576;
    int frame_size = 640;
    int offset = 0;
    struct sensevoice_state_t *state = (struct sensevoice_state_t *)ggml_handle->state;
    if(1)
    {
        memcpy(&input_matrix[0]->data_fp[0], state->chunk_cache, 64 * sizeof(float));
        memcpy(&input_matrix[0]->data_fp[64], &input_matrix[1]->data_fp[0], chunk_size * sizeof(float));
        for(int j = context_size; j < frame_size; j++)
        {
            input_matrix[0]->data_fp[j] = input_matrix[0]->data_fp[2 * context_size - j - 2];
        }
        memcpy(state->chunk_cache,  &input_matrix[1]->data_fp[chunk_size - 64], 64 * sizeof(float));
    }
#endif
    return SUCCESS;
}

int sensevoice_backend_process(struct ggml_handle_t *ggml_handle, float speech_prob)
{
#if 0
    int sample_rate = 16000;
    struct sensevoice_params_t *params = (struct sensevoice_params_t *)ggml_handle->params;
    struct sensevoice_state_t *state = (struct sensevoice_state_t *)ggml_handle->state;

    int chunk_size = 512;
    //to do 
    static bool  triggered = false;
    static int32_t temp_end = 0;
    static int32_t prev_end = 0, next_start = 0;
    static int32_t current_speech_start = 0, current_speech_end = 0;


    int32_t min_speech_samples = sample_rate * params->min_speech_duration_ms / 1000;
    int32_t speech_pad_samples = sample_rate * params->speech_pad_ms / 1000;
    int32_t max_speech_samples = sample_rate * params->max_speech_duration_ms / 1000 - chunk_size - 2 * speech_pad_samples;
    int32_t min_silence_samples = sample_rate * params->min_silence_duration_ms / 1000;
    int32_t min_silence_samples_at_max_speech = sample_rate * 98 / 1000;

    state->frame_count += 512;
    int i = state->frame_count;

    if(speech_prob >= params->threshold)
    {
        if(temp_end) 
            temp_end = 0;
        if(next_start < prev_end)
            next_start = i;
    }

    if(speech_prob >= params->threshold && !triggered)
    {
        triggered = true;
        current_speech_start = i;
        return SUCCESS;
    }

    if(speech_prob < params->neg_threshold && triggered)
    {
        if(temp_end == 0)
        {
            temp_end = i;
        }

        if(i - temp_end > min_silence_samples_at_max_speech)
        {
            prev_end = temp_end;
        }
        else
        {
            return SUCCESS;
        }

        if(i - prev_end < min_silence_samples)
        {
            return SUCCESS;
        }
        else
        {
            current_speech_end = prev_end;
            if(current_speech_end - current_speech_start > min_speech_samples)
            {
#if 0
                model->result.start_time = current_speech_start / (sample_rate * 1.0);
                model->result.end_time = current_speech_end / (sample_rate * 1.0);
                if(ggml_handle->cb)
                {
                    ggml_handle->cb(&model->result, sizeof(model->result), ggml_handle->user_data);
                }

#endif
                //pcm + current_speech_start, current_speech_end;
                printf("[%.2f-%.2f] \n", current_speech_start / (sample_rate * 1.0), current_speech_end / (sample_rate * 1.0));
                current_speech_end = current_speech_start = 0;
            }
            prev_end = next_start = 0;
            triggered = false;
        }
    }
#endif
    return SUCCESS;
}


int sensevoice_inference(struct ggml_handle_t *ggml_handle, matrix_t **input_matrix,
                        matrix_t **output_matrix)
{
#if 0
    struct sensevoice_params_t *params = (struct sensevoice_params_t *)ggml_handle->params;
    struct sensevoice_model_t *model = (struct sensevoice_model_t *)ggml_handle->model;
    struct sensevoice_state_t *state = (struct sensevoice_state_t *)ggml_handle->state;


    auto & sched = state->sched;
    ggml_cgraph *gf = sensevoice_build_cgraph(model, state);
    if (!ggml_backend_sched_alloc_graph(sched, gf))
    {
        // should never happen as we pre-allocate the memory
        LOG_ERROR("sensevoice_inference ggml_backend_sched_alloc_graph error");
        return ERROR;
    }

    sensevoice_frontend_process(ggml_handle, input_matrix);

    // set the input
    struct ggml_tensor *data = ggml_graph_get_tensor(gf, "audio_chunk");
    ggml_backend_tensor_set(data, input_matrix[0]->data, 0, ggml_nbytes(data));

    struct ggml_tensor *in_lstm_context = ggml_graph_get_tensor(gf, "in_lstm_context");
    struct ggml_tensor *in_lstm_hidden_state = ggml_graph_get_tensor(gf, "in_lstm_hidden_state");
    ggml_backend_tensor_copy(state->vad_lstm_context, in_lstm_context);
    ggml_backend_tensor_copy(state->vad_lstm_hidden_state, in_lstm_hidden_state);

    if (!ggml_graph_compute_helper(sched, gf, ggml_handle->n_thread))
    {
        LOG_ERROR("sensevoice_inference ggml_graph_compute_helper error");
        return ERROR;
    }

    // save output state
    struct ggml_tensor *lstm_context = ggml_graph_get_tensor(gf, "out_lstm_context");
    ggml_backend_tensor_copy(lstm_context, state->vad_lstm_context);
    struct ggml_tensor *lstm_hidden_state = ggml_graph_get_tensor(gf, "out_lstm_hidden_state");
    ggml_backend_tensor_copy(lstm_hidden_state, state->vad_lstm_hidden_state);

    float speech_prob = 0.0;

    ggml_backend_tensor_get(ggml_graph_get_tensor(gf, "logit"), &speech_prob, 0, sizeof(speech_prob));

    sensevoice_backend_process(ggml_handle, speech_prob);

#endif
    return SUCCESS;
}

