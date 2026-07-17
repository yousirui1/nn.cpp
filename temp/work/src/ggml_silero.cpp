#include "base.h"
#include <functional>
#include "matrix.h"

#include "gguf_loader.h"
#include "ggml_module.h"
#include "ggml_model.h"
#include "ggml_silero.h"
#include "ggml_sensevoice.h"
#include "ggml_vad.h"

#define SILERO_MAX_NODES 8192
#define VAD_CHUNK_SIZE 640


struct silero_model_t 
{
    STFT stft;
    std::vector<Conv1d> encoders;
    LSTM rnn;
    Conv1d decoder;
    float chunk_cache[64];
    int frame_count;

#if 0
    bool triggered = false;
    int32_t temp_end = 0;
    int32_t prev_end = 0, next_start = 0;
    int32_t current_speech_start = 0, current_speech_end = 0;
#endif

    struct vad_result_t result;
};

struct silero_state_t
{
    ggml_backend_sched_t sched = nullptr;
    std::vector<uint8_t> meta;

    struct ggml_tensor * vad_lstm_hidden_state;
    struct ggml_tensor * vad_lstm_context;
};

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
        

    }

    // encoder
    {
        {

            cur = ggml_conv_1d(ctx0, model->encoders[0].weight, cur, 1, 1, 1);
            cur = ggml_add(ctx0, cur, ggml_cont(ctx0, ggml_transpose(ctx0, model->encoders[0].bias)));
   

            cur = ggml_relu(ctx0, cur);

            //LOG_DEBUG("=================");
           cur = ggml_conv_1d(ctx0, model->encoders[1].weight, cur, 2, 1, 1);
           cur = ggml_add(ctx0, cur,  ggml_cont(ctx0, ggml_transpose(ctx0, model->encoders[1].bias)));

            cur = ggml_relu(ctx0, cur);

            // LOG_DEBUG("=================");


            cur = ggml_conv_1d(ctx0, model->encoders[2].weight, cur, 2, 1, 1);
            cur = ggml_add(ctx0, cur,  ggml_cont(ctx0, ggml_transpose(ctx0, model->encoders[2].bias)));
            cur = ggml_relu(ctx0, cur);
            // LOG_DEBUG("=================");
            cur = ggml_conv_1d(ctx0, model->encoders[3].weight, cur, 1, 1, 1);

            cur = ggml_add(ctx0, cur,  ggml_cont(ctx0, ggml_transpose(ctx0, model->encoders[3].bias)));
            cur = ggml_relu(ctx0, cur);
            //LOG_DEBUG("=================");
#if 0

            //cur = ggml_conv_1d(ctx0, model->encoders[0].weight, cur, 1, 1, 1);
            //cur = ggml_add(ctx0, cur, ggml_cont(ctx0, ggml_transpose(ctx0, model->encoders[0].bias)));
            //LOG_DEBUG("=================");
            ggml_backend_op_capabilities capabilities;
            //capabilities.im2col_f16 = 1;
            capabilities.im2col_f16 = 1;
            capabilities.repeat_f16 = 0;
            capabilities.concat_i32 = 1;

            //cur = ggml_conv_1d(ctx0, model->encoders[0].weight, cur, 1, 1, 1);
            //cur = ggml_add(ctx0, cur, ggml_cont(ctx0, ggml_transpose(ctx0, model->encoders[0].bias)));

            cur = model->encoders[0].build_cgraph(ctx0, cur, 1, 1, 1, 1, capabilities);

            //LOG_DEBUG("=================");
            cur = ggml_relu(ctx0, cur);

             //LOG_DEBUG("=================");
            //cur = ggml_conv_1d(ctx0, model->encoders[1].weight, cur, 2, 1, 1);
            //cur = ggml_add(ctx0, cur,  ggml_cont(ctx0, ggml_transpose(ctx0, model->encoders[1].bias)));

            cur = model->encoders[1].build_cgraph(ctx0, cur, 2, 1, 1, 1, capabilities);
            cur = ggml_relu(ctx0, cur);

            // LOG_DEBUG("=================");
            cur = model->encoders[2].build_cgraph(ctx0, cur, 2, 1, 1, 1, capabilities);

            //cur = ggml_conv_1d(ctx0, model->encoders[2].weight, cur, 2, 1, 1);
            //cur = ggml_add(ctx0, cur,  ggml_cont(ctx0, ggml_transpose(ctx0, model->encoders[2].bias)));
            cur = ggml_relu(ctx0, cur);
           // LOG_DEBUG("=================");
            cur = model->encoders[3].build_cgraph(ctx0, cur, 1, 1, 1, 1, capabilities);
            //cur = ggml_conv_1d(ctx0, model->encoders[3].weight, cur, 1, 1, 1);
            //cur = ggml_add(ctx0, cur,  ggml_cont(ctx0, ggml_transpose(ctx0, model->encoders[3].bias)));
            cur = ggml_relu(ctx0, cur);
            //LOG_DEBUG("=================");
#endif
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

}

int load_silero_model(struct ggml_handle_t *ggml_handle, const char *model_path,
                        shape_t *input_shape, int *in_nodes,shape_t *output_shape, int *out_nodes)
{
    struct ggml_context *ctx;
    ggml_backend_buffer_t buffer = nullptr;

    ggml_backend_t backend;
    ggml_backend_t cpu_backend;

    struct silero_params_t *params = new silero_params_t;
    if(NULL == params)
    {
        LOG_ERROR("silero params error %s", strerror(errno));
        return ERROR;
    }

    get_silero_default_params(params);

    struct silero_model_t *model = new silero_model_t; 
    if(NULL == model)
    {
        LOG_ERROR("silero malloc error %s", strerror(errno));
        delete params;
        return ERROR;
    }
    memset(model->chunk_cache, 0, sizeof(model->chunk_cache));
    model->frame_count = 0;

    struct silero_state_t *state = new silero_state_t; 
    if(NULL == model)
    {
        LOG_ERROR("silero state error %s", strerror(errno));
        delete model;
        delete params;
        return ERROR;
    }

    gguf_loader loader(model_path);

    ctx = ggml_init(ggml_init_params{ ggml_graph_overhead() * GGML_DEFAULT_GRAPH_SIZE, nullptr, true });

    backend = ggml_backend_init_best();
    cpu_backend = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr);

    std::string prefix = "_model";

    model->stft.OnLoad(loader, prefix);

    model->encoders.resize(params->n_encoder_layer);
    for(int i = 0; i < params->n_encoder_layer; ++i)
    {
        auto& layer = model->encoders[i];
        std::string name = prefix + ".encoder."  + std::to_string(i) + ".reparam_conv";
        layer.OnLoad(loader, name);
    }
    model->rnn.OnLoad(loader, prefix + ".decoder");
    model->decoder.OnLoad(loader, prefix + ".decoder.decoder.2");

    auto tensors = model->stft.get_all_tensors();

    for(auto &layer : model->encoders)
        for(auto &kv : layer.get_all_tensors())
            tensors.insert(std::move(kv));

    for(auto &kv : model->rnn.get_all_tensors())
        tensors.insert(std::move(kv));

    for(auto &kv : model->decoder.get_all_tensors())
        tensors.insert(std::move(kv));

    auto buft = ggml_backend_get_default_buffer_type(backend);
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
        ggml_backend_tensor_set_async(backend, tensor, data, 0, size);
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

    ggml_backend_t backends[] = { backend, cpu_backend };
    state->sched = ggml_backend_sched_new(
        backends,
        nullptr,
        backend == cpu_backend ? 1 : 2,
        GGML_DEFAULT_GRAPH_SIZE,
        true,
        true
    );

    ggml_cgraph *gf = silero_build_cgraph(model, state);
    ggml_backend_sched_alloc_graph(state->sched, gf);
    set_graph_backend(gf, state->sched, backend, nullptr, -1);
    ggml_backend_sched_reset(state->sched);

    state->vad_lstm_context = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, params->lstm_state_dim);
    ggml_backend_tensor_alloc(buffer, state->vad_lstm_context, buffer_base);
    buffer_base += get_aligned_size(state->vad_lstm_context->nb[1], alignment);

    state->vad_lstm_hidden_state = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, params->lstm_state_dim);
    ggml_backend_tensor_alloc(buffer, state->vad_lstm_hidden_state, buffer_base);
    buffer_base += get_aligned_size(state->vad_lstm_hidden_state->nb[1], alignment);

    ggml_backend_synchronize(backend);

    ggml_handle->params = (void *)params;
    ggml_handle->model = (void *)model;
    ggml_handle->state = (void *)state;

    *in_nodes = 2;
    set_shape(&input_shape[0], TENSOR_FLOAT32, 1, 640);
    set_shape(&input_shape[1], TENSOR_FLOAT32, 1, 512);

    *out_nodes = 1;
    set_shape(&output_shape[0], TENSOR_FLOAT32, 1, params->max_speech_duration_ms * 16);

    //ggml_print_tensor(model->encoders[0].weight);
    //ggml_print_tensor(model->encoders[0].bias);
    //exit(0);


    return SUCCESS;
}

int silero_frontend_process(struct ggml_handle_t *ggml_handle)
{
    int chunk_size = 512;
    int context_size = 576;
    int frame_size = 640;
    int offset = 0;
    struct silero_model_t *model = (struct silero_model_t *)ggml_handle->model;
    if(1)
    {
        memcpy(&ggml_handle->input_matrix[0]->data_fp[0], model->chunk_cache, 64 * sizeof(float));
        memcpy(&ggml_handle->input_matrix[0]->data_fp[64], &ggml_handle->input_matrix[1]->data_fp[0], chunk_size * sizeof(float));
        for(int j = context_size; j < frame_size; j++)
        {
            ggml_handle->input_matrix[0]->data_fp[j] = ggml_handle->input_matrix[0]->data_fp[2 * context_size - j - 2];     
        }
        memcpy(model->chunk_cache,  &ggml_handle->input_matrix[1]->data_fp[chunk_size - 64], 64 * sizeof(float));
    }

    return SUCCESS;
}

int silero_backend_process(struct ggml_handle_t *ggml_handle, float speech_prob)
{
    int sample_rate = 16000;
    struct silero_params_t *params = (struct silero_params_t *)ggml_handle->params;
    struct silero_model_t *model = (struct silero_model_t *)ggml_handle->model;

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

    model->frame_count += 512;
    int i = model->frame_count;

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
                model->result.start_time = current_speech_start / (sample_rate * 1.0);
                model->result.end_time = current_speech_end / (sample_rate * 1.0);
                if(ggml_handle->cb)
                {
                    ggml_handle->cb(&model->result, sizeof(model->result), ggml_handle->user_data);
                }

                //pcm + current_speech_start, current_speech_end;
                //printf("[%.2f-%.2f] \n", current_speech_start / (sample_rate * 1.0), current_speech_end / (sample_rate * 1.0));
                current_speech_end = current_speech_start = 0;
            }
            prev_end = next_start = 0;
            triggered = false;
        }
    }
    return SUCCESS;
}


int silero_inference(struct ggml_handle_t *ggml_handle, matrix_t **input_matrix, 
                        matrix_t **output_matrix, const int n_threads, void *param)
{
    struct silero_params_t *params = (struct silero_params_t *)ggml_handle->params;
    struct silero_model_t *model = (struct silero_model_t *)ggml_handle->model;
    struct silero_state_t *state = (struct silero_state_t *)ggml_handle->state;

    auto & sched = state->sched;
    ggml_cgraph *gf = silero_build_cgraph(model, state);
    if (!ggml_backend_sched_alloc_graph(sched, gf)) 
    {
        // should never happen as we pre-allocate the memory
        LOG_ERROR("silero_inference ggml_backend_sched_alloc_graph error");
        return ERROR;
    }

    silero_frontend_process(ggml_handle);

    // set the input
    struct ggml_tensor *data = ggml_graph_get_tensor(gf, "audio_chunk");
    ggml_backend_tensor_set(data, input_matrix[0]->data, 0, ggml_nbytes(data));

    struct ggml_tensor *in_lstm_context = ggml_graph_get_tensor(gf, "in_lstm_context");
    struct ggml_tensor *in_lstm_hidden_state = ggml_graph_get_tensor(gf, "in_lstm_hidden_state");
    ggml_backend_tensor_copy(state->vad_lstm_context, in_lstm_context);
    ggml_backend_tensor_copy(state->vad_lstm_hidden_state, in_lstm_hidden_state);

    if (!ggml_graph_compute_helper(sched, gf, n_threads)) 
    {
        LOG_ERROR("silero_inference ggml_graph_compute_helper error");
        return ERROR;
    }

    // save output state
    struct ggml_tensor *lstm_context = ggml_graph_get_tensor(gf, "out_lstm_context");
    ggml_backend_tensor_copy(lstm_context, state->vad_lstm_context);
    struct ggml_tensor *lstm_hidden_state = ggml_graph_get_tensor(gf, "out_lstm_hidden_state");
    ggml_backend_tensor_copy(lstm_hidden_state, state->vad_lstm_hidden_state);

    float speech_prob = 0.0;

    ggml_backend_tensor_get(ggml_graph_get_tensor(gf, "logit"), &speech_prob, 0, sizeof(speech_prob));
#if 0
    float encoder_out[4 * 128] = {0};

    ggml_backend_tensor_get(ggml_graph_get_tensor(gf, "output_test"), &encoder_out, 0, sizeof(encoder_out));

    //ggml_backend_tensor_get(ggml_graph_get_tensor(gf, "logit"), &speech_prob, 0, sizeof(speech_prob));

    FILE *fp = fopen("value.log", "w");
    for(int i = 0; i < 4 * 128; i++)
    {   
        if(i % 10 == 0)
        {
            printf("\n");
            fprintf(fp, "\n");
        }

        printf("%.2f ", encoder_out[i]);
        fprintf(fp, "%.2f ", encoder_out[i]);
    }
    printf("\n");
    fprintf(fp, "\n");
    fclose(fp);
#endif
    LOG_DEBUG("speech_prob %.2f ", speech_prob);

    if(param)
    {
        LOG_DEBUG("speech_prob %.2f ", speech_prob);
    }   
    silero_backend_process(ggml_handle, speech_prob);

    // /exit(0);
    return SUCCESS;

}




