#include "base.h"
#include "nn.h"
#include "gguf_loader.h"
#include "ggml_module.h"
#include "cosyvoice.h"
#include <random>
#include <chrono>
#include <thread>

void cosyvoice_default_params(cosyvoice_context_params_t *params)
{
    params->flow_use_flash_attn = true;
    params->llm_use_flash_attn = true;

    params->llm_allow_kv_cache_fallback = true;
    params->inference_buffer_policy = NN_INFERENCE_BUFFER_POLICY_BALANCED;

    params->n_batch = 256;
    params->n_max_seq = 2048;

    std::random_device rd;
    if(rd.entropy() == 0)
        params->seed = static_cast<uint32_t>(std::chrono::system_clock::now().time_since_epoch().count());
    else
        params->seed = rd();

    //params->builtin_sampler_rng_policy = NN_BUILTIN_SAMPLER_RNG_POLICY_RESET_PER_SESSION;
    //sampler = nullpptr;
    //sampler_ctx = nullpptr;
}

cosyvoice_context_t load_cosyvoice_model(const char *filename,
                                const cosyvoice_context_params_t params,
                                ggml_backend_t backend,
                                uint32_t n_threads,
                                uint32_t reserved)
{
    gguf_loader loader(filename);

    if(!loader)
        return nullptr;

    auto ctx = new cosyvoice_context(params, backend ? backend : ggml_backend_init_best());
    ctx->cosyvoice_model::load(loader);

    auto ggml_backend_set_n_threads = reinterpret_cast<ggml_backend_set_n_threads_t>(ggml_backend_reg_get_proc_address(ggml_backend_dev_backend_reg(ggml_backend_get_device(ctx->cpu_backend.get())), "ggml_backend_set_n_threads"));

    if (n_threads == 0)
        n_threads = std::thread::hardware_concurrency();
    if (n_threads != 0)
        ggml_backend_set_n_threads(ctx->cpu_backend.get(), n_threads);

    return ctx;
}

void init_cosyvoice_backend()
{
    ggml_time_init();
    ggml_log_set(nn_log_callback_default, nullptr);

    ggml_init_params params = {};
    ggml_context *ctx = ggml_init(params);
    ggml_free(ctx);
}

cosyvoice_model::cosyvoice_model(ggml_backend_t backend, const cosyvoice_context_params_t& params)
    : backend(backend), params(params), cpu_backend(backend),
    ctx(ggml_init(ggml_init_params{ .mem_size = ggml_graph_overhead() * GGML_DEFAULT_GRAPH_SIZE, .no_alloc = true })),
    ctx0(ggml_init(ggml_init_params{ .mem_size = ggml_graph_overhead() * GGML_DEFAULT_GRAPH_SIZE, .no_alloc = true })),
    ctx1(ggml_init(ggml_init_params{ .mem_size = ggml_tensor_overhead() * 4, .no_alloc = true })),
    gf(nullptr), llm_input(nullptr), llm_probs(nullptr), position_ids(nullptr), causal_mask(nullptr), kv_cache(nullptr),
    status(GGML_STATUS_SUCCESS)//, prompt_crc32(0), rand_noise_len(0) //to do 

{
    if (GGML_BACKEND_DEVICE_TYPE_CPU != ggml_backend_dev_type(ggml_backend_get_device(backend)))
    {
        cpu_backend.release();
        cpu_backend.reset(ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr));
    }

    empty_buffer_cache();
#if 0
    set_sampler(params.sampler, params.sampler_ctx);
    set_noise_callback(nullptr, nullptr);
#endif

    ggml_tensor* a = ggml_new_tensor_1d(ctx0.get(), GGML_TYPE_I32, 1);
    a = ggml_concat(ctx0.get(), a, a, 0);
    op_caps.concat_i32 = ggml_backend_supports_op(backend, a);

    a = ggml_new_tensor_3d(ctx0.get(), GGML_TYPE_F16, 11, 4, 51);
    a = ggml_repeat_4d(ctx0.get(), a, 11, 4, 51, 4);
    op_caps.repeat_f16 = ggml_backend_supports_op(backend, a);
    LOG_DEBUG("params->inference_buffer_policy %d", params.inference_buffer_policy);
}

void cosyvoice_model::empty_buffer_cache()
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
#if 0
    rand_noise.reset();
    rand_noise_len = 0;
#endif

    if (kv_cache)
        kv_cache->clear_offloaded_cache();
}

void cosyvoice_model::reset_shared_buffer(ggml_backend_buffer *new_buffer)
{
    token2wav_buffer.reset(new_buffer);
    if (params.inference_buffer_policy != NN_INFERENCE_BUFFER_POLICY_DEDICATED)
    {   
        kv_buffer.reset(new_buffer);
        params.n_max_seq = kv_cache->reset_buffer(new_buffer);
    }  
}

void cosyvoice_model::load(gguf_loader &loader)
{
    kv_type = this->params.llm_kv_cache_type;

    flow.OnLoad(loader, {});
    hift.OnLoad(loader, {});
    llm.OnLoad(loader, {});

    auto tensors = llm.get_all_tensors();
    for(auto &kv : flow.get_all_tensors())
        tensors.insert(std::move(kv));

    for(auto &kv : hift.get_all_tensors())
        tensors.insert(std::move(kv));

    ggml_init_params params = 
    {
        .mem_size = (tensors.size() + 8) * ggml_tensor_overhead()
        + ggml_backend_buft_get_alloc_size(ggml_backend_cpu_buffer_type(), llm.embed_tokens_weight)
        + ggml_backend_buft_get_alloc_size(ggml_backend_cpu_buffer_type(), llm.speech_embedding_weight),
        .mem_buffer = nullptr,
        .no_alloc = true
    };

    // init sinusodal position embedding
    constexpr int dim = 256;
    constexpr int half_dim = dim / 2;

    auto emb_buffer = std::make_unique<float[]>(half_dim);
    const auto emb = std::log(10000.f) / (half_dim - 1);

    for (int i = 0; i != half_dim; ++i)
        emb_buffer[i] = std::exp(i * -emb) * 1000;

    auto buft = ggml_backend_get_default_buffer_type(backend.get());
    auto alignment = ggml_backend_buft_get_alignment(buft);

    const int mel_dim = flow.decoder.estimator.mel_dim;
    size_t mem_size = get_aligned_size(sizeof(float) * half_dim, alignment)
        + get_aligned_size(sizeof(float) * (hift.nfft / 2 + 1), alignment)
        + get_aligned_size(sizeof(float) * hift.nfft, alignment)
        + get_aligned_size(sizeof(int) * this->params.n_batch, alignment)
        + get_aligned_size(sizeof(float) * (hift.nfft / 2 + 1) * hift.nfft, alignment) * 2
        + get_aligned_size((this->params.n_max_seq - 1) * this->params.n_batch * sizeof(ggml_fp16_t), alignment);
    for (const auto& [name, tensor] : tensors)
        if (tensor != &llm.embed_tokens_weight
            && tensor != &llm.speech_embedding_weight)
            mem_size += get_aligned_size(ggml_backend_buft_get_alloc_size(buft, *tensor), alignment);

    buffer.reset(ggml_backend_buft_alloc_buffer(buft, mem_size));
    auto buffer_base = reinterpret_cast<char*>(ggml_backend_buffer_get_base(buffer.get()));
    ctx.reset(ggml_init(params));

    auto set_tensor = [&](ggml_tensor* tensor, const void* data, size_t size)
    {   
        ggml_backend_tensor_set_async(backend.get(), tensor, data, 0, size);
        buffer_base += get_aligned_size(ggml_backend_buft_get_alloc_size(buft, tensor), alignment);
    };
    flow.decoder.estimator.time_embed.time_embed.emb = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_F32, half_dim);
    ggml_backend_tensor_alloc(buffer.get(), flow.decoder.estimator.time_embed.time_embed.emb, buffer_base);
    set_tensor(flow.decoder.estimator.time_embed.time_embed.emb, emb_buffer.get(), sizeof(float) * half_dim);

    for (const auto& [name, tensor] : tensors)
    {
        size_t tensor_size = ggml_nbytes(*tensor);
        if (tensor == &llm.embed_tokens_weight
            || tensor == &llm.speech_embedding_weight)
        {
            ggml_set_no_alloc(ctx.get(), false);
            auto new_tensor = ggml_new_tensor(ctx.get(), (*tensor)->type, GGML_MAX_DIMS, (*tensor)->ne);
            memcpy(new_tensor->data, (**tensor).data, tensor_size);
            *tensor = new_tensor;
            ggml_set_no_alloc(ctx.get(), true);
        }
        else
        {
            auto new_tensor = ggml_new_tensor(ctx.get(), (*tensor)->type, GGML_MAX_DIMS, (*tensor)->ne);
            ggml_backend_tensor_alloc(buffer.get(), new_tensor, buffer_base);
            set_tensor(new_tensor, ggml_get_data(*tensor), tensor_size);
            *tensor = new_tensor;
        }
        ggml_set_param(*tensor);
        ggml_set_name(*tensor, name.c_str());
    }

	//to do 
    //noise_rng.seed(this->params.seed);
    //sampler_rng.seed(noise_rng());

    hift.m_source.l_sin_gen.rand_ini = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_F32, hift.nfft / 2 + 1);
    ggml_backend_tensor_alloc(buffer.get(), hift.m_source.l_sin_gen.rand_ini, buffer_base);
    buffer_base += get_aligned_size(hift.m_source.l_sin_gen.rand_ini->nb[1], alignment);

    auto temp_buffer = std::make_unique<float[]>(hift.nfft);
    temp_buffer[0] = 0.f;

    //to do 
    //std::normal_distribution<float> dist(0.0f, 1.0f);
    //for (auto& i : std::span(temp_buffer.get() + 1, hift.nfft / 2))
    //    i = dist(noise_rng);

    //hift.set_rand_ini(temp_buffer.get());

    hift.window = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_F32, hift.nfft);
    for (int i = 0; i != hift.nfft; ++i)
        temp_buffer[i] = (1.0f - std::cos(2.0f * 3.14159265358979323846f * i / hift.nfft)) / 2.f;
    ggml_backend_tensor_alloc(buffer.get(), hift.window, buffer_base);
    set_tensor(hift.window, temp_buffer.get(), hift.nfft * sizeof(float));

    position_ids = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_I32, this->params.n_batch);
    ggml_set_name(position_ids, "position_ids");
    ggml_backend_tensor_alloc(buffer.get(), position_ids, buffer_base);
    full_position_ids.reset(new int[this->params.n_max_seq]);
    for (int i = 0; i != this->params.n_max_seq; ++i)
        full_position_ids.get()[i] = i;
    buffer_base += get_aligned_size(position_ids->nb[1], alignment);

    hift.fctx = create_fft_context(hift.nfft);
    hift.ictx = create_istft_context(hift.nfft, ctx.get(),
        [&](ggml_tensor* tensor, void* data, size_t size)
        {
            ggml_backend_tensor_alloc(buffer.get(), tensor, buffer_base);
            set_tensor(tensor, data, size);
        });

    causal_mask_buffer.reset(new ggml_fp16_t[(this->params.n_max_seq - 1) * this->params.n_batch]);
    causal_mask = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F16, this->params.n_max_seq - 1, this->params.n_batch);
    ggml_set_name(causal_mask, "attention_mask");
    ggml_backend_tensor_alloc(buffer.get(), causal_mask, buffer_base);

    int64_t id = gguf_find_key(loader, "stop_token_ids");
    auto stop_tok_data = reinterpret_cast<const int*>(gguf_get_arr_data(loader, id));
    id = static_cast<int64_t>(gguf_get_arr_n(loader, id));
    stop_tokens.insert(stop_tok_data, stop_tok_data + id);

#if 0
    //to do 
    config.temperature = 1.f;
    config.max_token_text_ratio = 20.f;
    config.min_token_text_ratio = 2.f;

    id = gguf_find_key(loader, "cosyvoice.instruction_prefix");
    if (id != -1)
    {
        auto str = gguf_get_val_str(loader, id);
        auto len = strlen(str) + 1;
        instruction_prefix.reset(new char[len]);
        memcpy(instruction_prefix.get(), str, len);
    }

    auto& sampling = config.sampling;
    LOAD_METADATA_NOPREFIX(sampling.top_k);
    LOAD_METADATA_NOPREFIX(sampling.top_p);
    LOAD_METADATA_NOPREFIX(sampling.win_size);
    LOAD_METADATA_NOPREFIX(sampling.tau_r);

    LOAD_METADATA_NOPREFIX(sample_rate);

	nucleus_probs.reset(new float[sampling.top_k * 2 + 1]);
#endif
	probs.reset(new float[llm.llm_decoder.weight->ne[1]]);
	batch_buffer.reset(new char[this->params.n_batch * std::max(llm.embed_tokens_weight->nb[1], llm.speech_embedding_weight->nb[1])]);

	if (this->params.llm_use_flash_attn)
		if (this->params.llm_allow_kv_cache_fallback
			&& ggml_is_quantized(kv_type))
		{
			auto cur_type = kv_type;
			while (cur_type != GGML_TYPE_F32)
			{
				auto q = ggml_new_tensor_3d(ctx0.get(), GGML_TYPE_F32, llm.layers[0].self_attn.q_proj.weight->ne[1] / llm.num_attention_heads, 1, llm.num_attention_heads);
				auto k = ggml_new_tensor_3d(ctx0.get(), kv_type, llm.layers[0].self_attn.k_proj.weight->ne[1] / llm.num_key_value_heads, 1, llm.num_key_value_heads);
				auto v = ggml_new_tensor_3d(ctx0.get(), kv_type, llm.layers[0].self_attn.v_proj.weight->ne[1] / llm.num_key_value_heads, 1, llm.num_key_value_heads);
				auto o = ggml_flash_attn_ext(ctx0.get(), q, k, v, nullptr, 1.f / std::sqrt(static_cast<float>(k->ne[0])), 0.f, 0.f);
				if (ggml_backend_supports_op(backend.get(), o))
				{
					kv_type = cur_type;
					break;
				}
				--reinterpret_cast<int&>(cur_type);
			}

			this->params.llm_use_flash_attn = cur_type != GGML_TYPE_F32;
            this->params.llm_kv_cache_type = kv_type;
		}
		else
		{
			auto q = ggml_new_tensor_3d(ctx0.get(), GGML_TYPE_F32, llm.layers[0].self_attn.q_proj.weight->ne[1] / llm.num_attention_heads, 1, llm.num_attention_heads);
			auto k = ggml_new_tensor_3d(ctx0.get(), kv_type, llm.layers[0].self_attn.k_proj.weight->ne[1] / llm.num_key_value_heads, 1, llm.num_key_value_heads);
			auto v = ggml_new_tensor_3d(ctx0.get(), kv_type, llm.layers[0].self_attn.v_proj.weight->ne[1] / llm.num_key_value_heads, 1, llm.num_key_value_heads);
			auto o = ggml_flash_attn_ext(ctx0.get(), q, k, v, nullptr, 1.f / std::sqrt(static_cast<float>(k->ne[0])), 0.f, 0.f);
			this->params.llm_use_flash_attn = ggml_backend_supports_op(backend.get(), o);
		}
	else if (ggml_is_quantized(kv_type))
		kv_type = GGML_TYPE_F16;

	kv_cache = new llm_kv_cache(
		backend.get(),
		kv_buffer,
		static_cast<int>(llm.layers.size()),
		static_cast<int>(llm.layers[0].self_attn.k_proj.weight->ne[1] / llm.num_key_value_heads),
		static_cast<int>(llm.layers[0].self_attn.v_proj.weight->ne[1] / llm.num_key_value_heads),
		llm.num_attention_heads,
		llm.num_key_value_heads,
		this->params.n_max_seq,
		kv_type,
		this->params.llm_use_flash_attn
	);

	switch (this->params.inference_buffer_policy)
	{
	    case NN_INFERENCE_BUFFER_POLICY_BALANCED:
	    case NN_INFERENCE_BUFFER_POLICY_SHARED:
		    token2wav_buffer.reset(kv_buffer.get());
	    case NN_INFERENCE_BUFFER_POLICY_DEDICATED:
		    break;
	    default:
		    throw std::invalid_argument("unexpected policy");
	}

	if (this->params.flow_use_flash_attn)
	{
		int heads = flow.decoder.estimator.transformer_blocks[0].attn.heads;
		auto q = ggml_new_tensor_4d(ctx0.get(), GGML_TYPE_F32, flow.decoder.estimator.transformer_blocks[0].attn.to_q.weight->ne[1] / heads, 1, heads, 2);
		auto k = ggml_new_tensor_4d(ctx0.get(), GGML_TYPE_F32, flow.decoder.estimator.transformer_blocks[0].attn.to_k.weight->ne[1] / heads, 1, heads, 2);
		auto v = ggml_new_tensor_4d(ctx0.get(), GGML_TYPE_F32, flow.decoder.estimator.transformer_blocks[0].attn.to_v.weight->ne[1] / heads, 1, heads, 2);
		auto o = ggml_flash_attn_ext(ctx0.get(), q, k, v, nullptr, 1.f / std::sqrt(static_cast<float>(k->ne[0])), 0.f, 0.f);
		this->params.flow_use_flash_attn = ggml_backend_supports_op(backend.get(), o);
	}

	for (auto& block : flow.decoder.estimator.transformer_blocks)
		block.attn.fattn = this->params.flow_use_flash_attn;

	orig_max_seq_len = this->params.n_max_seq;
	ggml_backend_synchronize(backend.get());
}


#if 0
bool cosyvoice_model::llm_job(const int *text, uint32_t text_len)
{

}

bool cosyvoice_model::token2wav()
{

}
#endif

#if 0

bool cosyvoice_model::tts_zero_shot()
{


}

bool cosyvoice_model::tts_instruct()
{


}

bool cosyvoice_model::tts_cross_lingual()
{

}


int cosyvoice_model::inference()
{

}


#endif


#if 0
uint32_t cosyvoice_model::llm_get_kv_cache_len()
{
    return kv_cache->cur_len;
}

bool cosyvoice_model::llm_set_kv_cache_len(uint32_t len)
{
    auto& cur_len = kv_cache->cur_len;
    if (len <= cur_len)
    {
        cur_len = len;
        return true;
    }
    return false;
}

uint32_t cosyvoice_model::llm_get_kv_cache_len()
{
    return kv_cache->cur_len;
}
#endif
