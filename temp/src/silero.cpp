#include "base.h"
#include "silero.h"
#include "gguf_loader.h"
#include <thread>

static void set_graph_backend(ggml_cgraph* gf, ggml_backend_sched_t sched, ggml_backend_t backend, ggml_backend_t cpu_backend, int nodes = -1)
{
    if (nodes < 0)
        nodes = ggml_graph_n_nodes(gf);

    for (int i = 0; i != nodes; ++i) {
        auto node = ggml_graph_node(gf, i);
        if (node->op == GGML_OP_CUSTOM)
        {
            do {
                ggml_backend_sched_set_tensor_backend(sched, node, cpu_backend);
                if (++i == nodes) return;
                node = ggml_graph_node(gf, i);
            } while ((node->op == GGML_OP_VIEW || node->op == GGML_OP_PERMUTE));

            goto set_main_backend;
        }
        else
        {
        set_main_backend:
            ggml_backend_sched_set_tensor_backend(sched, node, backend);
        }
    }
}



void silero_default_params(silero_context_params_t *params)
{
    params->n_encoder_layer = 4;
    params->lstm_state_dim = 128;

    params->n_batch = 256;
    
    //speech len
}


silero_context_t load_silero_model(const char *filename,
                                const silero_context_params_t params,
                                ggml_backend_t backend,
                                uint32_t n_threads,
                                uint32_t reserved)
{
    gguf_loader loader(filename);

    if(!loader)
        return nullptr;

    auto ctx = new silero_context(params, backend ? backend : ggml_backend_init_best());
    ctx->silero_model::load(loader);

    auto ggml_backend_set_n_threads = reinterpret_cast<ggml_backend_set_n_threads_t>(ggml_backend_reg_get_proc_address(ggml_backend_dev_backend_reg(ggml_backend_get_device(ctx->cpu_backend.get())), "ggml_backend_set_n_threads"));

    if (n_threads == 0)
        n_threads = std::thread::hardware_concurrency();
    if (n_threads != 0)
        ggml_backend_set_n_threads(ctx->cpu_backend.get(), n_threads);

    ctx->silero_model::build_graph();

    ctx->silero_model::inference();
    return ctx;
}

void init_silero_backend()
{
    ggml_time_init();
    ggml_log_set(nn_log_callback_default, nullptr);

    ggml_init_params params = {};
    ggml_context *ctx = ggml_init(params);
    ggml_free(ctx);
}

silero_model::silero_model(ggml_backend_t backend, const silero_context_params_t& params)
    : backend(backend), params(params), cpu_backend(backend),
    ctx(ggml_init(ggml_init_params{ .mem_size = ggml_graph_overhead() * GGML_DEFAULT_GRAPH_SIZE, .no_alloc = true })),
    gf(nullptr), lstm_context(nullptr), lstm_hidden_state(nullptr), 
    status(GGML_STATUS_SUCCESS)//, prompt_crc32(0), rand_noise_len(0) //to do 
{

    if (GGML_BACKEND_DEVICE_TYPE_CPU != ggml_backend_dev_type(ggml_backend_get_device(backend)))
    {
        cpu_backend.release();
        cpu_backend.reset(ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr));
    }

    empty_buffer_cache();

    ggml_tensor* a = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_I32, 1);
    a = ggml_concat(ctx.get(), a, a, 0);
    op_caps.concat_i32 = ggml_backend_supports_op(backend, a);

    a = ggml_new_tensor_3d(ctx.get(), GGML_TYPE_F16, 11, 4, 51);
    a = ggml_repeat_4d(ctx.get(), a, 11, 4, 51, 4);
    op_caps.repeat_f16 = ggml_backend_supports_op(backend, a);
}

void silero_model::empty_buffer_cache()
{
    ggml_backend_t backends[] = { backend.get(), cpu_backend.get() };
    sched.reset(ggml_backend_sched_new(
        backends,
        nullptr,
        backend == cpu_backend ? 1 : 2,
        GGML_DEFAULT_GRAPH_SIZE,
        true,
        true
    ));
}

void silero_model::load(gguf_loader &loader)
{
    //model.OnLoad(loader, "_model");
    //stft

    std::string prefix = "_model";

    stft.OnLoad(loader, prefix);

    //metadata
    encoders.resize(this->params.n_encoder_layer);  
    for(int i = 0; i < this->params.n_encoder_layer; ++i)
    {
        auto& layer = encoders[i];
        std::string name = prefix + ".encoder."  + std::to_string(i) + ".reparam_conv";
        layer.OnLoad(loader, name);
    }
    rnn.OnLoad(loader, prefix + ".decoder");
    decoder.OnLoad(loader, prefix + ".decoder.decoder.2");

    auto tensors = stft.get_all_tensors();

    for(auto &layer : encoders)
        for(auto &kv : layer.get_all_tensors())
            tensors.insert(std::move(kv));

    for(auto &kv : rnn.get_all_tensors())
        tensors.insert(std::move(kv));

    for(auto &kv : decoder.get_all_tensors())
        tensors.insert(std::move(kv));

#if 0
    //int graph_size = 2048;
    ggml_init_params params = 
    {
        .mem_size = (tensors.size() + 2048) * ggml_tensor_overhead(),
        .mem_buffer = nullptr,
        .no_alloc = true
    };
#endif

    auto buft = ggml_backend_get_default_buffer_type(backend.get());
    auto alignment = ggml_backend_buft_get_alignment(buft);
    size_t mem_size = get_aligned_size(sizeof(float) * this->params.lstm_state_dim * 2, alignment); 
    //LOG_DEBUG("mem_size %lld ", mem_size);

    for(const auto &[name, tensor] : tensors)
    {
        mem_size += get_aligned_size(ggml_backend_buft_get_alloc_size(buft, *tensor), alignment);
    }

    LOG_DEBUG("mem_size %lld ", mem_size);

    buffer.reset(ggml_backend_buft_alloc_buffer(buft, mem_size));
    auto buffer_base = reinterpret_cast<char*>(ggml_backend_buffer_get_base(buffer.get()));

    //ctx.reset(ggml_init(params)); //to do 

    auto set_tensor = [&](ggml_tensor* tensor, const void* data, size_t size)
    {   
        ggml_backend_tensor_set_async(backend.get(), tensor, data, 0, size);
        buffer_base += get_aligned_size(ggml_backend_buft_get_alloc_size(buft, tensor), alignment);
    };  

    for (const auto& [name, tensor] : tensors)
    {   
        size_t tensor_size = ggml_nbytes(*tensor);
        {
            auto new_tensor = ggml_new_tensor(ctx.get(), (*tensor)->type, GGML_MAX_DIMS, (*tensor)->ne);
            ggml_backend_tensor_alloc(buffer.get(), new_tensor, buffer_base);
            set_tensor(new_tensor, ggml_get_data(*tensor), tensor_size);
            *tensor = new_tensor;
        }
        ggml_set_param(*tensor);
        ggml_set_name(*tensor, name.c_str());
    }

    
    lstm_context = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_F32, this->params.lstm_state_dim);
    ggml_backend_tensor_alloc(buffer.get(), lstm_context, buffer_base);
    buffer_base += get_aligned_size(lstm_context->nb[1], alignment);

    lstm_hidden_state = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_F32, this->params.lstm_state_dim);
    ggml_backend_tensor_alloc(buffer.get(), lstm_hidden_state, buffer_base);
    buffer_base += get_aligned_size(lstm_hidden_state->nb[1], alignment);

    ggml_backend_synchronize(backend.get());
}

void silero_model::build_graph()
{
    ggml_tensor *cur = nullptr;
    auto ctx0 = ctx.get();

    ggml_reset(ctx0);

    ggml_backend_sched_reset(sched.get());

    gf = ggml_new_graph(ctx0);

    input = ggml_new_tensor_1d(ctx0, GGML_TYPE_F32, 640); 
    ggml_set_name(input, "speech chunk");
    ggml_set_input(input);

    // stft
    cur = stft.build_cgraph(ctx0, input);

    // encoder
    {
        cur = ggml_conv_1d(ctx0, encoders[0].weight, cur, 1, 1, 1); 
        cur = ggml_add(ctx0, cur, ggml_cont(ctx0, ggml_transpose(ctx0, encoders[0].bias)));
        cur = ggml_relu(ctx0, cur);

        cur = ggml_conv_1d(ctx0, encoders[1].weight, cur, 2, 1, 1);
        cur = ggml_add(ctx0, cur,  ggml_cont(ctx0, ggml_transpose(ctx0, encoders[1].bias)));
        cur = ggml_relu(ctx0, cur);

        cur = ggml_conv_1d(ctx0, encoders[2].weight, cur, 2, 1, 1);
        cur = ggml_add(ctx0, cur,  ggml_cont(ctx0, ggml_transpose(ctx0, encoders[2].bias)));
        cur = ggml_relu(ctx0, cur);

        cur = ggml_conv_1d(ctx0, encoders[3].weight, cur, 1, 1, 1);
        cur = ggml_add(ctx0, cur,  ggml_cont(ctx0, ggml_transpose(ctx0, encoders[3].bias)));
        cur = ggml_relu(ctx0, cur);
    }

    // decoder
    {
        cur = rnn.build_cgraph(ctx0, cur);

        cur = ggml_relu(ctx0, cur);
        cur = ggml_conv_1d(ctx0, decoder.weight, ggml_cont(ctx0, ggml_transpose(ctx0, cur)), 1, 0, 1);
        cur = ggml_add(ctx0, cur, ggml_transpose(ctx0, decoder.bias));
        ggml_set_name(cur, "decoder_out");
        cur = ggml_sigmoid(ctx0, cur);
        ggml_set_name(cur, "logit");
    }

    ggml_build_forward_expand(gf, cur);
    set_graph_backend(gf, sched.get(), backend.get(), nullptr);

    ggml_backend_sched_synchronize(sched.get());
    ggml_backend_sched_alloc_graph(sched.get(), gf);
    LOG_DEBUG("ggml_graph_n_nodes %d ", ggml_graph_n_nodes(gf));
}

int silero_model::inference()
{
 	struct ggml_tensor *in_lstm_context = ggml_graph_get_tensor(gf, "in_lstm_context");
    struct ggml_tensor *in_lstm_hidden_state = ggml_graph_get_tensor(gf, "in_lstm_hidden_state");

    ggml_backend_tensor_copy(lstm_context,  in_lstm_context);
    ggml_backend_tensor_copy(lstm_hidden_state, in_lstm_hidden_state);

    ggml_backend_sched_graph_compute(sched.get(), gf);

    struct ggml_tensor *out_lstm_context = ggml_graph_get_tensor(gf, "out_lstm_context");
    struct ggml_tensor *out_lstm_hidden_state = ggml_graph_get_tensor(gf, "out_lstm_hidden_state");
    ggml_backend_tensor_copy(out_lstm_context, lstm_context);
    ggml_backend_tensor_copy(out_lstm_hidden_state, lstm_hidden_state);
}
