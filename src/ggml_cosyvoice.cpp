#include "base.h"
#include "gguf_loader.h"
#include "ggml_model.h"
#include "ggml_module.h"
#include "ggml_cosyvoice.h"
#include "span_compat.h"
#include <random>
#include <chrono>
#include <map>

struct ggml_sched_t
{
    ggml_backend_sched_t sched = nullptr;
    std::vector<uint8_t> meta;
};

struct cosyvoice_state_t
{
    //worker
    ggml_context *ctx;
    ggml_backend_buffer_t buffer;
    ggml_backend_buffer_t cpu_buffer;       //fft 

    ggml_sched_t flow_sched;
    ggml_sched_t hift_sched;
    ggml_sched_t llm_sched;

    std::vector<ggml_backend_t> backends;


    std::mt19937 noise_rng;

};


int init_ggml()
{

}

int ggml_model_weight_alloc(ggml_context *ctx, ggml_backend_t dev_backend, ggml_backend_buffer_t &buffer, const std::map<std::string, ggml_tensor**> &tensors)
{
    auto buft = ggml_backend_get_default_buffer_type(dev_backend);
    auto alignment = ggml_backend_buft_get_alignment(buft);
    size_t mem_size = 0;

    for(const auto &[name, tensor] : tensors)
    {
        mem_size += get_aligned_size(ggml_backend_buft_get_alloc_size(buft, *tensor), alignment);
    }

    buffer = ggml_backend_buft_alloc_buffer(buft, mem_size);
    auto buffer_base = reinterpret_cast<char*>(ggml_backend_buffer_get_base(buffer));

    ggml_backend_buffer_set_usage(buffer, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);

    auto set_tensor = [&](ggml_tensor* tensor, const void* data, size_t size)
    {
        ggml_backend_tensor_set_async(dev_backend, tensor, data, 0, size);
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
    return mem_size;
}

#define SENSEVOICE_MAX_NODE 8192

static bool init_ggml_sched_graph(struct ggml_sched_t &allocr, 
                    std::vector<ggml_backend_t> backends,
                    std::function<struct ggml_cgraph *()> &&get_graph)
{
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

void get_cosyvoice_default_params(struct cosyvoice_params_t *params)
{
    params->flow_use_flash_attn = true;
    params->llm_use_flash_attn = true;

    //k v kv_fallback separate_buffer
    //params->llm_kv_cache_type = GGML_TYPE_Q8_0;

    params->llm_allow_kv_cache_fallback = true; // to do

    //buffer_policy

    params->n_batch = 256;
    params->n_llm_max_seq = 2048;

    std::random_device rd;
    if (rd.entropy() == 0)
        params->seed = static_cast<uint32_t>(std::chrono::system_clock::now().time_since_epoch().count());
    else 
        params->seed = rd();

    params->n_workers = 1;
    // sampler
    // sampler_ctx
};


void* load_cosyvoice_prompt_speech(const char *weight_path)
{
    gguf_loader gguf(weight_path);
    if (!gguf) 
        return nullptr;

    ggml_tensor *feat = ggml_get_tensor(gguf, "feat");
    ggml_tensor *embedding = ggml_get_tensor(gguf, "embedding");
    ggml_tensor *tokens = ggml_get_tensor(gguf, "tokens");
    ggml_tensor *text = ggml_get_tensor(gguf, "text");

    if (!feat || !embedding || !tokens || !text
        || !ggml_is_matrix(feat) || !ggml_is_matrix(embedding) || !ggml_is_vector(tokens) || !ggml_is_vector(text)
        || feat->type != GGML_TYPE_F32 || embedding->type != GGML_TYPE_F32 || tokens->type != GGML_TYPE_I32 || text->type != GGML_TYPE_I8)
    {
        errno = EILSEQ;
        return nullptr;
    }

#if 0
    auto prompt = new cosyvoice_prompt_speech;
    prompt->feat = matrix{ uint32_t(feat->ne[1]), uint32_t(feat->ne[0]) };
    prompt->embedding = matrix{ uint32_t(embedding->ne[1]), uint32_t(embedding->ne[0]) };
    prompt->tokens = std::make_pair(std::shared_ptr<int>(new int[tokens->ne[0]], std::default_delete<int[]>()), static_cast<uint32_t>(tokens->ne[0]));
    prompt->text = std::make_pair(std::shared_ptr<char>(new char[text->ne[0]], std::default_delete<char[]>()), static_cast<uint32_t>(text->ne[0]));

    memcpy(prompt->feat.data, feat->data, feat->nb[2]);
    memcpy(prompt->embedding.data, embedding->data, embedding->nb[2]);
    memcpy(prompt->tokens.first.get(), tokens->data, tokens->nb[1]);
    memcpy(prompt->text.first.get(), text->data, text->nb[1]);
    prompt->calculate_crc32();

    uint32_t crc32;
    if (gguf.get_metadata("crc32", crc32)
        && crc32 != prompt->crc32)
    {
        cosyvoice_prompt_speech_free(prompt);
        errno = EILSEQ;
        return nullptr;
    }
    return prompt;
#endif
}

int init_cosyvoice_state(cosyvoice_model_t *model, cosyvoice_state_t *state, cosyvoice_params_t *hparams, ggml_backend_t backend, ggml_backend_t cpu_backend)
{
    constexpr int dim = 256;
    constexpr int half_dim = dim / 2;

    auto& flow = model->flow;
    auto& hift = model->hift;
    auto& llm = model->llm;

    ggml_init_params params = {
         .mem_size = (6 + 2 * hparams->n_workers) * ggml_tensor_overhead(),
         .no_alloc = true,
    };

    state->ctx = ggml_init(params);
    if(!state->ctx)
    {
        LOG_ERROR("");
        return ERROR;
    }

    auto buft = ggml_backend_get_default_buffer_type(backend);
    auto alignment = ggml_backend_buft_get_alignment(buft);
    
    size_t mem_size = get_aligned_size(sizeof(float) * half_dim, alignment)
            + get_aligned_size(sizeof(float) * (hift.nfft / 2 + 1), alignment)
            + get_aligned_size(sizeof(float) * (hift.nfft / 2 + 1) * hift.nfft, alignment) * 2
            + get_aligned_size(sizeof(float) * (hift.nfft / 2 + 1), alignment)
            + (get_aligned_size(sizeof(int) * hparams->n_batch, alignment) + 

                get_aligned_size((hparams->n_llm_max_seq - 1) * hparams->n_batch * sizeof(ggml_fp16_t), alignment)
              ) * hparams->n_workers;
            

    state->buffer = ggml_backend_buft_alloc_buffer(buft, mem_size);
    auto buffer_base = reinterpret_cast<char*>(ggml_backend_buffer_get_base(state->buffer));

    auto set_tensor = [&](ggml_tensor* tensor, const void* data, size_t size)
    {
        ggml_backend_tensor_set_async(backend, tensor, data, 0, size);
        buffer_base += get_aligned_size(ggml_backend_buft_get_alloc_size(buft, tensor), alignment);
    };

    mem_size = ggml_backend_buft_get_alloc_size(ggml_backend_cpu_buffer_type(), llm.embed_tokens_weight)
        + ggml_backend_buft_get_alloc_size(ggml_backend_cpu_buffer_type(), llm.speech_embedding_weight)
        + sizeof(float) * hift.nfft;

    state->cpu_buffer = ggml_backend_buft_alloc_buffer(ggml_backend_get_default_buffer_type(cpu_backend), mem_size);
    auto cpu_buffer_base = reinterpret_cast<char*>(ggml_backend_buffer_get_base(state->cpu_buffer));

    auto set_cpu_tensor = [&](ggml_tensor* tensor, const void* data, size_t size)
    {
        ggml_backend_tensor_set(tensor, data, 0, size);
        cpu_buffer_base += ggml_backend_buft_get_alloc_size(ggml_backend_get_default_buffer_type(cpu_backend), tensor);
    };

    std::unique_ptr<float[]> emb_buffer(new float[half_dim]);
    const auto emb = std::log(10000.f) / (half_dim - 1);

    for (int i = 0; i != half_dim; ++i)
        emb_buffer[i] = std::exp(i * -emb) * 1000;

    flow.decoder.estimator.time_embed.time_embed.emb = ggml_new_tensor_1d(state->ctx, GGML_TYPE_F32, half_dim);
    ggml_backend_tensor_alloc(state->buffer, flow.decoder.estimator.time_embed.time_embed.emb, buffer_base);
    set_tensor(flow.decoder.estimator.time_embed.time_embed.emb, emb_buffer.get(), sizeof(float) * half_dim);

    hift.window = ggml_new_tensor_1d(state->ctx, GGML_TYPE_F32, hift.nfft);
    ggml_set_param(hift.window);
    ggml_set_name(hift.window, "stft_window"); 
    ggml_backend_tensor_alloc(state->cpu_buffer, hift.window, cpu_buffer_base);

    for (int i = 0; i != hift.nfft; ++i)
        reinterpret_cast<float*>(hift.window->data)[i] = (1.0f - std::cos(2.0f * 3.14159265358979323846f * i / hift.nfft)) / 2.f;

    //to do 
    hift.fctx = create_fft_context(hift.nfft);
    hift.ictx = create_istft_context(hift.nfft, state->ctx,
        [&](ggml_tensor* tensor, void* data, size_t size)
        {
            ggml_backend_tensor_alloc(state->buffer, tensor, buffer_base);
            set_tensor(tensor, data, size);
            ggml_set_param(tensor);
            ggml_backend_synchronize(backend);
        });


    for(uint32_t i = 0; i != hparams->n_workers; ++i)
    {
#if 0
        auto worker = workers + i;
        
        worker->position_ids = ggml_new_tensor_1d(shared->ctx.get(), GGML_TYPE_I32, shared->params.n_batch);
        ggml_set_name(worker->position_ids, ("position_ids." + std::to_string(i)).c_str());
        ggml_backend_tensor_alloc(shared->buffer.get(), worker->position_ids, buffer_base);
        ggml_set_param(worker->position_ids);
        worker->full_position_ids.reset(new int[shared->params.n_max_seq]);
        for (int i = 0; i != shared->params.n_max_seq; ++i)
            worker->full_position_ids.get()[i] = i;
        buffer_base += get_aligned_size(worker->position_ids->nb[1], alignment);
        
        worker->causal_mask_buffer.reset(new ggml_fp16_t[(shared->params.n_max_seq - 1) * shared->params.n_batch]);
        worker->causal_mask = ggml_new_tensor_2d(shared->ctx.get(), GGML_TYPE_F16, shared->params.n_max_seq - 1, shared->params.n_batch);
        ggml_set_name(worker->causal_mask, ("attention_mask." + std::to_string(i)).c_str());
        ggml_set_param(worker->causal_mask);
        ggml_backend_tensor_alloc(shared->buffer.get(), worker->causal_mask, buffer_base);
        buffer_base += get_aligned_size(worker->causal_mask->nb[0] * worker->causal_mask->nb[1], alignment);


        worker.nucleus_probs_capacity = static_cast<uint32_t>(sampling.top_k * 2);
        worker.nucleus_probs.reset(new float[worker.nucleus_probs_capacity]);
        worker.nucleus_probs_len = 0;
        worker.probs.reset(new float[llm.llm_decoder.weight->ne[1]]);
        worker.batch_buffer.reset(new char[shared->params.n_batch * std::max(llm.embed_tokens_weight->nb[1], llm.speech_embedding_weight->nb[1])]);

        worker.kv_cache.build_kv_cache(
            backend,
            worker.kv_buffer,
            static_cast<int>(llm.layers.size()),
            static_cast<int>(llm.layers[0].self_attn.k_proj.weight->ne[1] / llm.num_key_value_heads),
            static_cast<int>(llm.layers[0].self_attn.v_proj.weight->ne[1] / llm.num_key_value_heads),
            llm.num_attention_heads,
            llm.num_key_value_heads,
            shared->params.n_max_seq,
            cv3_shared->k_type,
            cv3_shared->v_type,
            shared->params.llm_use_flash_attn
        );
        //
#endif
    }

    state->noise_rng.seed(hparams->seed);
    hift.m_source.l_sin_gen.rand_ini = ggml_new_tensor_1d(state->ctx, GGML_TYPE_F32, hift.nb_harmonics + 1);
    ggml_backend_tensor_alloc(state->buffer, hift.m_source.l_sin_gen.rand_ini, buffer_base);

    std::unique_ptr<float[]> rand_ini_buffer(new float[hift.nb_harmonics + 1]);
    rand_ini_buffer[0] = 0.f;
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);
    for(auto &i : simple_span<float>(rand_ini_buffer.get() + 1, hift.nb_harmonics))
        i = dist(state->noise_rng);

    hift.set_rand_ini(rand_ini_buffer.get());

    ggml_backend_synchronize(backend);

    return SUCCESS;
}

ggml_cgraph *cosyvoice_build_cgraph_flow(cosyvoice_model_t *model, cosyvoice_state_t *state, cosyvoice_params_t *hparams)
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

ggml_cgraph *cosyvoice_build_cgraph_hift(cosyvoice_model_t *model, cosyvoice_state_t *state, cosyvoice_params_t *hparams)
{
#if 0
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
#endif
}

ggml_cgraph *cosyvoice_build_cgraph_llm(cosyvoice_model_t *model, cosyvoice_state_t *state, cosyvoice_params_t *hparams)
{
    return nullptr;
}

int load_cosyvoice_weight(gguf_loader &loader, cosyvoice_model_t *model, ggml_backend_t backend)
{
    auto& flow = model->flow;
    auto& hift = model->hift;
    auto& llm = model->llm;

    flow.onload(loader, {});
    hift.onload(loader, {});
    llm.onload(loader, {});

    int64_t id = gguf_find_key(loader, "stop_token_ids");
    auto stop_tok_data = reinterpret_cast<const int*>(gguf_get_arr_data(loader, id));
    id = static_cast<int64_t>(gguf_get_arr_n(loader, id));
    model->stop_tokens.insert(stop_tok_data, stop_tok_data + id);

    id = gguf_find_key(loader, "cosyvoice.instruction_prefix");
    if (id != -1)
    {   
        auto str = gguf_get_val_str(loader, id);
        auto len = strlen(str) + 1;
        model->instruction_prefix.reset(new char[len]);
        memcpy(model->instruction_prefix.get(), str, len);  //to do state
    }   

    auto arch = loader.get_string("general.architecture");
    model->architecture.reset(new char[arch.size() + 1]);
    memcpy(model->architecture.get(), arch.data(), arch.size() + 1);

    auto &sampling = model->sampling;
    LOAD_METADATA_NOPREFIX(sampling.top_k);
    LOAD_METADATA_NOPREFIX(sampling.top_p);
    LOAD_METADATA_NOPREFIX(sampling.win_size);
    LOAD_METADATA_NOPREFIX(sampling.tau_r);

    auto tensors = llm.get_all_tensors();
    for (auto& kv : flow.get_all_tensors())
        tensors.insert(std::move(kv));
    for (auto& kv : hift.get_all_tensors())
        tensors.insert(std::move(kv));

    //k_type
    //v_type
 
    model->ctx = ggml_init(ggml_init_params{ .mem_size = ggml_graph_overhead() * GGML_DEFAULT_GRAPH_SIZE, .no_alloc = true });

    ggml_model_weight_alloc(model->ctx, backend, model->buf_weights, tensors);
    return SUCCESS;
}


int load_cosyvoice_model(struct ggml_handle_t *ggml_handle, const char *model_data, int model_size)
{
    float* prompt_audio_data = nullptr;
    uint32_t prompt_audio_length = 0;
    uint32_t prompt_audio_sample_rate = 0;
    int max_llm_len = 2048;
    bool text_normalization_enabled = true;

    struct cosyvoice_params_t cosyvoice_params, *params;
    params = &cosyvoice_params;
    
    get_cosyvoice_default_params(params);

    const char *prompt_path = "";

    //load_cosyvoice_prompt_speech(prompt_path);

    struct cosyvoice_model_t cosyvoice_model, *model;
    model = &cosyvoice_model;

    gguf_loader loader(model_data);

    ggml_handle->backend = ggml_backend_init_best();
    ggml_handle->cpu_backend = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, NULL);

    load_cosyvoice_weight(loader, model, ggml_handle->backend);

    //load_cosyvoice_tokenizer();

    //cosyvoice_prompt_speech_handle prompt_speech;

    struct cosyvoice_state_t cosyvoice_state, *state;
    state = &cosyvoice_state;

    state->backends.push_back(ggml_handle->backend);
    state->backends.push_back(ggml_handle->cpu_backend);
    
    bool ret = init_ggml_sched_graph(state->flow_sched, state->backends, 
            [&] () { return  cosyvoice_build_cgraph_flow(model, state, params); });

    ret = init_ggml_sched_graph(state->hift_sched, state->backends, 
            [&] () { return  cosyvoice_build_cgraph_hift(model, state, params); });

    ret = init_ggml_sched_graph(state->flow_sched, state->backends, 
            [&] () { return  cosyvoice_build_cgraph_llm(model, state, params); });
}
