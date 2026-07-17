#include "cosyvoice_model.h"
#include "cosyvoice_llm_kv_cache.h"

#include <cstring>
#include "span_compat.h"
#include <chrono>
#include <mutex>

#define COSYVOICE_DEFAULT_LLM_MAX_SEQ_LEN 4096


namespace
{
#if defined(__APPLE__) && defined(__aarch64__)
constexpr bool backend_looks_uma(ggml_backend_t backend, ggml_backend_buffer* buffer) { return true; }
#else
constexpr size_t kUmaProbeBytes = 128ull * 1024ull * 1024ull;
constexpr size_t kUmaProbeMinBytes = 64ull * 1024ull * 1024ull;
constexpr int kUmaProbeIters = 4;

struct transfer_fit
{
    double bandwidth_bytes_per_us = 0.0;
    double overhead_us = 0.0;
};

static transfer_fit fit_transfer(size_t size_1, double time_1_us, size_t size_2, double time_2_us)
{
    transfer_fit result;
    if (size_1 <= size_2 || time_1_us <= time_2_us)
        return result;

    const auto bw = static_cast<double>(size_1 - size_2) / (time_1_us - time_2_us);
    if (bw <= 0.0)
        return result;

    result.bandwidth_bytes_per_us = bw;
    result.overhead_us = time_1_us - static_cast<double>(size_1) / bw;
    if (result.overhead_us < 0.0)
        result.overhead_us = 0.0;
    return result;
}

template <typename F>
static double measure_min_us(F&& fn)
{
    double min_us = (std::numeric_limits<double>::max)();
    for (int i = 0; i != kUmaProbeIters; ++i) {
        const auto start = ggml_time_us();
        fn();
        const auto elapsed = ggml_time_us() - start;
        if (elapsed > 0 && static_cast<double>(elapsed) < min_us)
            min_us = static_cast<double>(elapsed);
    }
    return min_us < (std::numeric_limits<double>::max)() ? min_us : 0.0;
}

bool backend_looks_uma(ggml_backend_t backend, ggml_backend_buffer* buffer)
{
    ggml_backend_dev_props props;
    ggml_backend_dev_get_props(ggml_backend_get_device(backend), &props);
    if (props.type == GGML_BACKEND_DEVICE_TYPE_IGPU)
        return true;
    if (strncmp(props.name, "Vulkan", 6) == 0 && props.type == GGML_BACKEND_DEVICE_TYPE_GPU)
        return false;

    constexpr auto size_1 = kUmaProbeBytes - (kUmaProbeBytes % sizeof(float));
    constexpr auto size_2 = kUmaProbeMinBytes - (kUmaProbeMinBytes % sizeof(float));
    if constexpr (size_1 == 0 || size_2 == 0 || size_1 <= size_2)
        return false;

    if (!backend || !buffer)
        return false;

    const auto buffer_size = ggml_backend_buffer_get_size(buffer);
    if (buffer_size < kUmaProbeBytes)
        return false;

    ggml_tensor tensor = {};
    tensor.type = GGML_TYPE_I8;
    tensor.ne[0] = static_cast<int64_t>(size_1);
    tensor.ne[1] = 1;
    tensor.ne[2] = 1;
    tensor.ne[3] = 1;
    tensor.nb[0] = 1;
    tensor.nb[1] = size_1;
    tensor.nb[2] = size_1;
    tensor.nb[3] = size_1;
    ggml_backend_tensor_alloc(buffer, &tensor, ggml_backend_buffer_get_base(buffer));
    std::unique_ptr<char[]> host_src(new char[size_1 * 2]);

    auto* src = host_src.get();
    auto* dst = src + size_1;

    memcpy(dst, src, size_1);

    const auto memcpy_1_us = measure_min_us([&] { memcpy(dst, src, size_1); });
    const auto memcpy_2_us = measure_min_us([&] { memcpy(dst, src, size_2); });

    ggml_backend_tensor_set(&tensor, src, 0, size_1);
    const auto backend_1_us = measure_min_us([&] { ggml_backend_tensor_set(&tensor, src, 0, size_1); });

    ggml_backend_tensor_set(&tensor, src, 0, size_2);
    const auto backend_2_us = measure_min_us([&] { ggml_backend_tensor_set(&tensor, src, 0, size_2); });

    if (memcpy_1_us <= 0.0 || memcpy_2_us <= 0.0 || backend_1_us <= 0.0 || backend_2_us <= 0.0)
        return false;

    const auto memcpy_fit = fit_transfer(size_1, memcpy_1_us, size_2, memcpy_2_us);
    const auto backend_fit = fit_transfer(size_1, backend_1_us, size_2, backend_2_us);
    if (memcpy_fit.bandwidth_bytes_per_us <= 0.0 || backend_fit.bandwidth_bytes_per_us <= 0.0)
        return false;

    constexpr double mib_per_us_to_mib_per_s = 1000000.0 / (1024.0 * 1024.0);
    const auto memcpy_mib_s = memcpy_fit.bandwidth_bytes_per_us * mib_per_us_to_mib_per_s;
    const auto backend_mib_s = backend_fit.bandwidth_bytes_per_us * mib_per_us_to_mib_per_s;
    const auto uma = backend_fit.bandwidth_bytes_per_us >= memcpy_fit.bandwidth_bytes_per_us * 0.7;
    char uma_buf[512];
    snprintf(uma_buf, sizeof(uma_buf),
        "UMA probe: memcpy=%.1f MiB/s (overhead %.1f us), backend=%.1f MiB/s (overhead %.1f us), guess: UMA=%s\n",
        memcpy_mib_s,
        memcpy_fit.overhead_us,
        backend_mib_s,
        backend_fit.overhead_us,
        uma ? "true" : "false");
    cosyvoice_call_ggml_log_callback(GGML_LOG_LEVEL_INFO, uma_buf);

    return uma;
}
#endif
}



static float* cosyvoice_default_noise_callback(
    cosyvoice_noise_callback_stage_t stage,
    uint32_t                         length,
    float* noise,
    cosyvoice_model_shared* shared
)
{
    switch (stage)
    {
    case COSYVOICE_NOISE_CALLBACK_STAGE_BEFORE_FLOW:
    case COSYVOICE_NOISE_CALLBACK_STAGE_BEFORE_HIFT:
    {
        std::shared_lock read_lock(shared->noise_mutex);
        if (length <= shared->rand_noise_len && shared->rand_noise)
            return shared->rand_noise.get();
    }
    {
        std::unique_lock write_lock(shared->noise_mutex);
        if (length > shared->rand_noise_len)
        {
            std::unique_ptr<float[]> new_noise(new float[length]);
            if (shared->rand_noise)
                memcpy(new_noise.get(), shared->rand_noise.get(), shared->rand_noise_len * sizeof(float));

            std::normal_distribution<float> dist(0.0f, 1.0f);
            for (auto& i : simple_span<float>(new_noise.get() + shared->rand_noise_len, length - shared->rand_noise_len))
                i = dist(shared->noise_rng);
            shared->rand_noise.swap(new_noise);
            shared->rand_noise_len = length;
        }
        return shared->rand_noise.get();
    }
    }
    return nullptr;
}

void cosyvoice_init_default_context_params(cosyvoice_context_params_t* params)
{
	params->flow_use_flash_attn = true;
	params->llm_use_flash_attn = true;

	params->llm_kv_cache_type = COSYVOICE_LLM_KV_CACHE_TYPE_Q8_0;
	params->llm_allow_kv_cache_fallback = true;
	params->inference_buffer_policy = COSYVOICE_INFERENCE_BUFFER_POLICY_BALANCED;

    params->n_batch = 256;
    params->n_max_seq = COSYVOICE_DEFAULT_LLM_MAX_SEQ_LEN;

    params->seed = cosyvoice_generate_random_seed();
    params->builtin_sampler_rng_policy = COSYVOICE_BUILTIN_SAMPLER_RNG_POLICY_RESET_PER_SESSION;

	params->sampler = nullptr;
	params->sampler_ctx = nullptr;
}

cosyvoice_model_shared::cosyvoice_model_shared(const cosyvoice_context_params_v2_cpp& params)
    : params(params), ctx(nullptr), backend_uma(false), rand_noise_len(0), noise_callback(nullptr), noise_callback_ctx(nullptr) {}

cosyvoice_worker_context::cosyvoice_worker_context(ggml_backend_t backend)
    : backend(backend), cpu_backend(backend),
    ctx0(ggml_init(ggml_init_params{ ggml_graph_overhead() * kCosyVoiceGraphSize, nullptr, true })),
    gf(nullptr), llm_input(nullptr), llm_probs(nullptr), position_ids(nullptr), causal_mask(nullptr), kv_cache(),
    status(GGML_STATUS_SUCCESS), prompt_crc32(0), sampler_seed(0), sampler(nullptr), sampler_ctx(nullptr), builtin_sampler_rng_policy(COSYVOICE_BUILTIN_SAMPLER_RNG_POLICY_RESET_PER_SESSION), nucleus_probs_capacity(0), nucleus_probs_len(0) {}

cosyvoice_model::cosyvoice_model(ggml_backend_t backend, const cosyvoice_context_params_v2_cpp& params)
    : shared(new cosyvoice_model_shared(params)), workers(reinterpret_cast<cosyvoice_worker_context*>(malloc(sizeof(cosyvoice_worker_context) * params.n_workers)))
{
    auto dev = ggml_backend_get_device(backend);
    bool dev_is_cpu = ggml_backend_dev_type(dev) == GGML_BACKEND_DEVICE_TYPE_CPU;

    shared->noise_rng.seed(shared->params.seed);
    shared->noise_callback = reinterpret_cast<cosyvoice_noise_callback_t>(cosyvoice_default_noise_callback);
    shared->noise_callback_ctx = shared;

    auto init_worker = [&](cosyvoice_worker_context* cur)
    {
        worker = new (cur) cosyvoice_worker_context(backend);
        if (!dev_is_cpu)
        {
            worker->cpu_backend.release();
            worker->cpu_backend.reset(ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr));
        }
        empty_buffer_cache();
        worker->config.temperature = 1.f;
        worker->config.max_token_text_ratio = 20.f;
        worker->config.min_token_text_ratio = 2.f;
        worker->builtin_sampler_rng_policy = shared->params.builtin_sampler_rng_policy;
        worker->sampler_seed = shared->noise_rng();
        worker->sampler_rng.seed(worker->sampler_seed);
        worker->sampler = reinterpret_cast<cosyvoice_sampler_t>(cosyvoice_llm_sampler);
        worker->sampler_ctx = workers;
    };

    init_worker(workers);
    for (size_t i = 1; i != params.n_workers; ++i)
    {
        backend = ggml_backend_dev_init(dev, nullptr);
        init_worker(workers + i);
    }

    set_sampler(params.sampler, params.sampler_ctx);
    worker->config.temperature = 1.f;
    worker->config.max_token_text_ratio = 20.f;
    worker->config.min_token_text_ratio = 2.f;

    auto ctx0 = worker->ctx0.get();
    auto& op_caps = shared->op_caps;
    ggml_tensor* a = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, 1);
    a = ggml_concat(ctx0, a, a, 0);
    op_caps.concat_i32 = ggml_backend_supports_op(backend, a);

    a = ggml_new_tensor_3d(ctx0, GGML_TYPE_F16, 11, 4, 51);
    a = ggml_repeat_4d(ctx0, a, 11, 4, 51, 4);
    op_caps.repeat_f16 = ggml_backend_supports_op(backend, a);

    a = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, 16, 4);
    a = ggml_pad(ctx0, a, 4, 0, 0, 0);
    op_caps.pad = ggml_backend_supports_op(backend, a);

    a = ggml_fill(ctx0, a, 0.0f);
    op_caps.fill = ggml_backend_supports_op(backend, a);

    a = ggml_cumsum(ctx0, a);
    op_caps.cumsum = ggml_backend_supports_op(backend, a);

    a = ggml_new_tensor_1d(ctx0, GGML_TYPE_F32, 16);
    a = ggml_pad_reflect_1d(ctx0, a, 1, 0);
    op_caps.pad_reflect_1d = ggml_backend_supports_op(backend, a);

    op_caps.top_k = ggml_backend_supports_op(backend, ggml_top_k(ctx0, a, 5));

    op_caps.leaky_relu = ggml_backend_supports_op(backend, ggml_leaky_relu(ctx0, a, 0.01f, false));

    op_caps.sin = ggml_backend_supports_op(backend, ggml_sin(ctx0, a));
    op_caps.cos = ggml_backend_supports_op(backend, ggml_cos(ctx0, a));

    op_caps.arange = ggml_backend_supports_op(backend, ggml_arange(ctx0, 0.f, 10.f, 1.f));

    op_caps.elu = ggml_backend_supports_op(backend, ggml_elu(ctx0, a));
    op_caps.abs = ggml_backend_supports_op(backend, ggml_abs(ctx0, a));
    op_caps.floor = ggml_backend_supports_op(backend, ggml_floor(ctx0, a));

    {
        auto a2 = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, 16, 4);
        auto b = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, 16, 4);
        op_caps.acc = ggml_backend_supports_op(backend, ggml_acc(ctx0, a2, b, a2->nb[1], a2->nb[2], a2->nb[3], 0));
    }

    {
        ggml_tensor* w = ggml_new_tensor_3d(ctx0, GGML_TYPE_F16, 3, 4, 8);
        a = ggml_new_tensor_3d(ctx0, GGML_TYPE_F16, 16, 4, 1);
        a = ggml_im2col(ctx0, w, a, 1, 0, 0, 0, 1, 0, false, GGML_TYPE_F16);
        op_caps.im2col_f16 = ggml_backend_supports_op(backend, a);
    }

    worker = workers;
}

bool cosyvoice_model::using_builtin_sampler() const
{
    return worker->sampler == reinterpret_cast<cosyvoice_sampler_t>(cosyvoice_llm_sampler);
}

bool cosyvoice_model::reset_builtin_sampler_rng()
{
    if (using_builtin_sampler())
    {
        worker->sampler_rng.seed(worker->sampler_seed);
        return true;
    }
    return false;
}

cosyvoice_builtin_sampler_rng_policy_t cosyvoice_model::get_builtin_sampler_rng_policy()
{
    return worker->builtin_sampler_rng_policy;
}

bool cosyvoice_model::set_sampler_seed(uint32_t seed)
{
    worker->sampler_seed = seed;
    return reset_builtin_sampler_rng();
}

uint32_t cosyvoice_model::get_sampler_seed()
{
    return worker->sampler_seed;
}

bool cosyvoice_model::set_builtin_sampler_rng_policy(cosyvoice_builtin_sampler_rng_policy_t policy)
{
	if (using_builtin_sampler())
		switch (policy)
		{
		case COSYVOICE_BUILTIN_SAMPLER_RNG_POLICY_RESET_PER_SESSION:
		case COSYVOICE_BUILTIN_SAMPLER_RNG_POLICY_CONTINUE_ACROSS_SESSIONS:
            worker->builtin_sampler_rng_policy = policy;
            return true;
		}
	return false;
}

cosyvoice_model::~cosyvoice_model()
{
    if (get_ref_count() == 1)
    {
        for (uint32_t i = 0; i != shared->params.n_workers; ++i)
        {
            auto worker = workers + i;

            if (worker->backend == worker->cpu_backend)
                worker->cpu_backend.release();

            switch (shared->params.inference_buffer_policy)
            {
            case COSYVOICE_INFERENCE_BUFFER_POLICY_BALANCED:
            case COSYVOICE_INFERENCE_BUFFER_POLICY_SHARED:
                worker->kv_buffer.release();
            }

            worker->~cosyvoice_worker_context();
        }

        delete shared;
        free(workers);
    }
}

void cosyvoice_model::empty_buffer_cache()
{
    ggml_backend_t backends[] = { worker->backend.get(), worker->cpu_backend.get() };
    worker->sched.reset(ggml_backend_sched_new(
        backends,
        nullptr,
        worker->backend == worker->cpu_backend ? 1 : 2,
        kCosyVoiceSchedGraphSize,
        true,
        true
    ));

    {
        std::unique_lock lock(shared->noise_mutex);
        shared->rand_noise.reset();
        shared->rand_noise_len = 0;
    }

    worker->kv_cache.clear_offloaded_cache();
}

static ggml_type cosyvoice_llm_kv_cache_type_to_ggml(cosyvoice_llm_kv_cache_type_t t)
{
    switch (t)
    {
    case COSYVOICE_LLM_KV_CACHE_TYPE_F32: return GGML_TYPE_F32;
    case COSYVOICE_LLM_KV_CACHE_TYPE_F16: return GGML_TYPE_F16;
    case COSYVOICE_LLM_KV_CACHE_TYPE_Q8_0: return GGML_TYPE_Q8_0;
    case COSYVOICE_LLM_KV_CACHE_TYPE_Q5_1: return GGML_TYPE_Q5_1;
    case COSYVOICE_LLM_KV_CACHE_TYPE_Q5_0: return GGML_TYPE_Q5_0;
    case COSYVOICE_LLM_KV_CACHE_TYPE_Q4_1: return GGML_TYPE_Q4_1;
    case COSYVOICE_LLM_KV_CACHE_TYPE_Q4_0: return GGML_TYPE_Q4_0;
    default: GGML_ABORT("unexpected kv cache type");
    }
}   
    
static cosyvoice_llm_kv_cache_type_t cosyvoice_ggml_to_llm_kv_cache_type(ggml_type t)
{   
    switch (t)
    {
    case GGML_TYPE_F32: return COSYVOICE_LLM_KV_CACHE_TYPE_F32;
    case GGML_TYPE_F16: return COSYVOICE_LLM_KV_CACHE_TYPE_F16;
    case GGML_TYPE_Q8_0: return COSYVOICE_LLM_KV_CACHE_TYPE_Q8_0;
    case GGML_TYPE_Q5_1: return COSYVOICE_LLM_KV_CACHE_TYPE_Q5_1;
    case GGML_TYPE_Q5_0: return COSYVOICE_LLM_KV_CACHE_TYPE_Q5_0;
    case GGML_TYPE_Q4_1: return COSYVOICE_LLM_KV_CACHE_TYPE_Q4_1;
    case GGML_TYPE_Q4_0: return COSYVOICE_LLM_KV_CACHE_TYPE_Q4_0;
    default: GGML_ABORT("unexpected ggml type for kv cache");
    }
}

static ggml_type cosyvoice_get_kv_fallback_type(ggml_type t)
{
    switch (t)
    {
    case GGML_TYPE_Q4_0: return GGML_TYPE_Q4_1;
    case GGML_TYPE_Q4_1: return GGML_TYPE_Q5_0;
    case GGML_TYPE_Q5_0: return GGML_TYPE_Q5_1;
    case GGML_TYPE_Q5_1: return GGML_TYPE_Q8_0;
    case GGML_TYPE_Q8_0: return GGML_TYPE_F16;
    case GGML_TYPE_F16:  return GGML_TYPE_F32;
    default: GGML_ABORT("fatal error");
    }
}

void cosyvoice_model_3::load(gguf_loader& loader)
{
    auto& flow = cv3_shared->flow;
    auto& hift = cv3_shared->hift;
    auto& llm = cv3_shared->llm;

    flow.onload(loader, {});
    hift.onload(loader, {});
    llm.onload(loader, {});

    auto tensors = llm.get_all_tensors();
    for (auto& kv : flow.get_all_tensors())
        tensors.insert(std::move(kv));
    for (auto& kv : hift.get_all_tensors())
        tensors.insert(std::move(kv));

    ggml_init_params params =
    {
        (tensors.size() + 6 + 2 * shared->params.n_workers) * ggml_tensor_overhead(),
        nullptr,
        true
    };

    // init sinusodal position embedding
    constexpr int dim = 256;
    constexpr int half_dim = dim / 2;

    std::unique_ptr<float[]> emb_buffer(new float[half_dim]);
    const auto emb = std::log(10000.f) / (half_dim - 1);

    for (int i = 0; i != half_dim; ++i)
        emb_buffer[i] = std::exp(i * -emb) * 1000;

    auto backend = worker->backend.get();
    auto cpu_backend = worker->cpu_backend.get();
    auto buft = ggml_backend_get_default_buffer_type(backend);
    auto alignment = ggml_backend_buft_get_alignment(buft);

    const int mel_dim = flow.decoder.estimator.mel_dim;
    size_t mem_size = get_aligned_size(sizeof(float) * half_dim, alignment)
        + get_aligned_size(sizeof(float) * (hift.nfft / 2 + 1), alignment)
        + get_aligned_size(sizeof(float) * (hift.nfft / 2 + 1) * hift.nfft, alignment) * 2
        + (get_aligned_size(sizeof(int) * shared->params.n_batch, alignment)
            + get_aligned_size((shared->params.n_max_seq - 1) * shared->params.n_batch * sizeof(ggml_fp16_t), alignment)) * shared->params.n_workers;
    for (const auto& [name, tensor] : tensors)
        if (tensor != &llm.embed_tokens_weight
            && tensor != &llm.speech_embedding_weight)
            mem_size += get_aligned_size(ggml_backend_buft_get_alloc_size(buft, *tensor), alignment);

    shared->buffer.reset(ggml_backend_buft_alloc_buffer(buft, mem_size));
    auto buffer_base = reinterpret_cast<char*>(ggml_backend_buffer_get_base(shared->buffer.get()));

    shared->backend_uma = backend_looks_uma(backend, shared->buffer.get());
    if (shared->params.inference_buffer_policy == COSYVOICE_INFERENCE_BUFFER_POLICY_BALANCED
        && shared->backend_uma)
    {
        shared->params.inference_buffer_policy = COSYVOICE_INFERENCE_BUFFER_POLICY_DEDICATED;
        cosyvoice_call_ggml_log_callback(GGML_LOG_LEVEL_INFO, "Detected UMA-like backend memory; switching balanced inference buffers to dedicated mode.\n");
    }

    shared->ctx.reset(ggml_init(params));

    mem_size = ggml_backend_buft_get_alloc_size(ggml_backend_cpu_buffer_type(), llm.embed_tokens_weight)
        + ggml_backend_buft_get_alloc_size(ggml_backend_cpu_buffer_type(), llm.speech_embedding_weight)
        + sizeof(float) * hift.nfft;
    shared->cpu_buffer.reset(ggml_backend_buft_alloc_buffer(ggml_backend_get_default_buffer_type(cpu_backend), mem_size));
    auto cpu_buffer_base = reinterpret_cast<char*>(ggml_backend_buffer_get_base(shared->cpu_buffer.get()));

    auto set_tensor = [&](ggml_tensor* tensor, const void* data, size_t size)
    {
        ggml_backend_tensor_set_async(backend, tensor, data, 0, size);
        buffer_base += get_aligned_size(ggml_backend_buft_get_alloc_size(buft, tensor), alignment);
    };
    auto set_cpu_tensor = [&](ggml_tensor* tensor, const void* data, size_t size)
    {
        ggml_backend_tensor_set(tensor, data, 0, size);
        cpu_buffer_base += ggml_backend_buft_get_alloc_size(ggml_backend_get_default_buffer_type(cpu_backend), tensor);
    };
    flow.decoder.estimator.time_embed.time_embed.emb = ggml_new_tensor_1d(shared->ctx.get(), GGML_TYPE_F32, half_dim);
    ggml_backend_tensor_alloc(shared->buffer.get(), flow.decoder.estimator.time_embed.time_embed.emb, buffer_base);
    set_tensor(flow.decoder.estimator.time_embed.time_embed.emb, emb_buffer.get(), sizeof(float) * half_dim);

    for (const auto& [name, tensor] : tensors)
    {
        size_t tensor_size = ggml_nbytes(*tensor);
        auto new_tensor = ggml_new_tensor(shared->ctx.get(), (*tensor)->type, GGML_MAX_DIMS, (*tensor)->ne);
        if (tensor == &llm.embed_tokens_weight
            || tensor == &llm.speech_embedding_weight)
        {
            ggml_backend_tensor_alloc(shared->cpu_buffer.get(), new_tensor, cpu_buffer_base);
            set_cpu_tensor(new_tensor, ggml_get_data(*tensor), tensor_size);
        }
        else
        {
            ggml_backend_tensor_alloc(shared->buffer.get(), new_tensor, buffer_base);
            set_tensor(new_tensor, ggml_get_data(*tensor), tensor_size);
        }

        *tensor = new_tensor;
        ggml_set_param(*tensor);
        ggml_set_name(*tensor, name.c_str());
    }

    hift.window = ggml_new_tensor_1d(shared->ctx.get(), GGML_TYPE_F32, hift.nfft);
    ggml_set_param(hift.window);
    ggml_set_name(hift.window, "stft_window");
    ggml_backend_tensor_alloc(shared->cpu_buffer.get(), hift.window, cpu_buffer_base);
    for (int i = 0; i != hift.nfft; ++i)
        reinterpret_cast<float*>(hift.window->data)[i] = (1.0f - std::cos(2.0f * 3.14159265358979323846f * i / hift.nfft)) / 2.f;

    hift.fctx = create_fft_context(hift.nfft);
    hift.ictx = create_istft_context(hift.nfft, shared->ctx.get(),
        [&](ggml_tensor* tensor, void* data, size_t size)
        {
            ggml_backend_tensor_alloc(shared->buffer.get(), tensor, buffer_base);
            set_tensor(tensor, data, size);
            ggml_set_param(tensor);
            ggml_backend_synchronize(backend);
        });

    for (uint32_t i = 0; i != shared->params.n_workers; ++i)
    {
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
    }

    shared->noise_rng.seed(shared->params.seed);

    hift.m_source.l_sin_gen.rand_ini = ggml_new_tensor_1d(shared->ctx.get(), GGML_TYPE_F32, hift.nb_harmonics + 1);
    ggml_backend_tensor_alloc(shared->buffer.get(), hift.m_source.l_sin_gen.rand_ini, buffer_base);

    std::unique_ptr<float[]> rand_ini_buffer(new float[hift.nb_harmonics + 1]);
    rand_ini_buffer[0] = 0.f;
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);
    for (auto& i : simple_span<float>(rand_ini_buffer.get() + 1, hift.nb_harmonics))
        i = dist(shared->noise_rng);
    hift.set_rand_ini(rand_ini_buffer.get());

    int64_t id = gguf_find_key(loader, "stop_token_ids");
    auto stop_tok_data = reinterpret_cast<const int*>(gguf_get_arr_data(loader, id));
    id = static_cast<int64_t>(gguf_get_arr_n(loader, id));
    cv3_shared->stop_tokens.insert(stop_tok_data, stop_tok_data + id);

    id = gguf_find_key(loader, "cosyvoice.instruction_prefix");
    if (id != -1)
    {
        auto str = gguf_get_val_str(loader, id);
        auto len = strlen(str) + 1;
        shared->instruction_prefix.reset(new char[len]);
        memcpy(shared->instruction_prefix.get(), str, len);
    }

    shared->config.temperature = 1.f;
    shared->config.max_token_text_ratio = 20.f;
    shared->config.min_token_text_ratio = 2.f;

    auto& sampling = shared->config.sampling;
    LOAD_METADATA_NOPREFIX(sampling.top_k);
    LOAD_METADATA_NOPREFIX(sampling.top_p);
    LOAD_METADATA_NOPREFIX(sampling.win_size);
    LOAD_METADATA_NOPREFIX(sampling.tau_r);

    for (uint32_t i = 0; i != shared->params.n_workers; ++i)
        workers[i].config = shared->config;

    if (shared->params.llm_use_flash_attn)
    {
        auto fattn_check = [&](ggml_type check_k, ggml_type check_v) -> bool
        {
            auto q = ggml_new_tensor_3d(worker->ctx0.get(), GGML_TYPE_F32, llm.layers[0].self_attn.q_proj.weight->ne[1] / llm.num_attention_heads, 1, llm.num_attention_heads);
            auto k = ggml_new_tensor_3d(worker->ctx0.get(), check_k, llm.layers[0].self_attn.k_proj.weight->ne[1] / llm.num_key_value_heads, 1, llm.num_key_value_heads);
            auto v = ggml_new_tensor_3d(worker->ctx0.get(), check_v, llm.layers[0].self_attn.v_proj.weight->ne[1] / llm.num_key_value_heads, 1, llm.num_key_value_heads);
            auto o = ggml_flash_attn_ext(worker->ctx0.get(), q, k, v, nullptr, 1.f / std::sqrt(static_cast<float>(k->ne[0])), 0.f, 0.f);
            return ggml_backend_supports_op(backend, o);
        };

        shared->params.llm_use_flash_attn = fattn_check(GGML_TYPE_F32, GGML_TYPE_F32);
        if (shared->params.llm_use_flash_attn)
        {
            if (shared->params.llm_allow_kv_cache_fallback)
            {
                ggml_type cur_type;

                do
                {
                    if (shared->params.llm_kv_cache_separate_buffers)
                    {
                        if (auto k_type = cosyvoice_llm_kv_cache_type_to_ggml(shared->params.llm_k_cache_type),
                            v_type = cosyvoice_llm_kv_cache_type_to_ggml(shared->params.llm_v_cache_type);
                            fattn_check(k_type, v_type))
                        {
                            cv3_shared->k_type = k_type;
                            cv3_shared->v_type = v_type;
                            shared->params.llm_kv_cache_type = COSYVOICE_MAKE_SEPARATE_KV_CACHE(
                                shared->params.llm_k_cache_type,
                                shared->params.llm_v_cache_type,
                                shared->params.llm_v_cache_type);
                            break;
                        }

                        cur_type = cosyvoice_llm_kv_cache_type_to_ggml(shared->params.llm_kv_cache_fallback);
                    }
                    else
                        cur_type = cosyvoice_llm_kv_cache_type_to_ggml(shared->params.llm_kv_cache_type);

                    do
                    {
                        if (fattn_check(cur_type, cur_type))
                        {
                            cv3_shared->k_type = cur_type;
                            cv3_shared->v_type = cur_type;
                            shared->params.llm_kv_cache_type = cosyvoice_ggml_to_llm_kv_cache_type(cur_type);
                            break;
                        }

                        cur_type = cosyvoice_get_kv_fallback_type(cur_type);
                    } while (cur_type != GGML_TYPE_F32);

                    if (shared->params.llm_kv_cache_separate_buffers)
                        shared->params.llm_kv_cache_type = COSYVOICE_MAKE_SEPARATE_KV_CACHE(
                            cosyvoice_ggml_to_llm_kv_cache_type(cv3_shared->k_type),
                            cosyvoice_ggml_to_llm_kv_cache_type(cv3_shared->v_type),
                            cosyvoice_ggml_to_llm_kv_cache_type(cv3_shared->v_type));
                } while (false);
            }
            else if (shared->params.llm_kv_cache_separate_buffers)
            {
                auto k_type = cosyvoice_llm_kv_cache_type_to_ggml(shared->params.llm_k_cache_type);
                auto v_type = cosyvoice_llm_kv_cache_type_to_ggml(shared->params.llm_v_cache_type);
                shared->params.llm_use_flash_attn = fattn_check(k_type, v_type);
                if (shared->params.llm_use_flash_attn)
                {
                    cv3_shared->k_type = k_type;
                    cv3_shared->v_type = v_type;
                    shared->params.llm_kv_cache_type = COSYVOICE_MAKE_SEPARATE_KV_CACHE(
                        shared->params.llm_k_cache_type,
                        shared->params.llm_v_cache_type,
                        shared->params.llm_v_cache_type);
                }
            }
            else
            {
                auto kv_type = cosyvoice_llm_kv_cache_type_to_ggml(shared->params.llm_kv_cache_type);
                shared->params.llm_use_flash_attn = fattn_check(kv_type, kv_type);
            }
        }
    }

    if (!shared->params.llm_use_flash_attn)
    {
        ggml_type cur_type;
        auto attn_check = [&](ggml_type check_k, ggml_type check_v) -> bool
        {
            if (ggml_is_quantized(check_v)) return false;
            auto q = ggml_new_tensor_3d(worker->ctx0.get(), GGML_TYPE_F32, llm.layers[0].self_attn.q_proj.weight->ne[1] / llm.num_attention_heads, 1, llm.num_attention_heads);
            auto k = ggml_new_tensor_3d(worker->ctx0.get(), check_k, llm.layers[0].self_attn.k_proj.weight->ne[1] / llm.num_key_value_heads, 1, llm.num_key_value_heads);
            auto v = ggml_new_tensor_3d(worker->ctx0.get(), check_v, 1, llm.layers[0].self_attn.v_proj.weight->ne[1] / llm.num_key_value_heads, llm.num_key_value_heads);
            auto s = ggml_mul_mat(worker->ctx0.get(), k, q);
            auto o = ggml_mul_mat(worker->ctx0.get(), v, s);
            return ggml_backend_supports_op(backend, o) && ggml_backend_supports_op(backend, s);
        };

        do
        {
            if (shared->params.llm_kv_cache_separate_buffers)
            {
                if (auto k_type = cosyvoice_llm_kv_cache_type_to_ggml(shared->params.llm_k_cache_type),
                    v_type = cosyvoice_llm_kv_cache_type_to_ggml(shared->params.llm_v_cache_type);
                    attn_check(k_type, v_type))
                {
                    cv3_shared->k_type = k_type;
                    cv3_shared->v_type = v_type;
                    shared->params.llm_kv_cache_type = COSYVOICE_MAKE_SEPARATE_KV_CACHE(
                        shared->params.llm_k_cache_type,
                        shared->params.llm_v_cache_type,
                        shared->params.llm_v_cache_type);
                    break;
                }
                else if (shared->params.llm_allow_kv_cache_fallback && attn_check(k_type, GGML_TYPE_F16))
                {
                    cv3_shared->k_type = cosyvoice_llm_kv_cache_type_to_ggml(shared->params.llm_k_cache_type);
                    cv3_shared->v_type = GGML_TYPE_F16;
                    shared->params.llm_kv_cache_type = COSYVOICE_MAKE_SEPARATE_KV_CACHE(
                        shared->params.llm_k_cache_type,
                        COSYVOICE_LLM_KV_CACHE_TYPE_F16,
                        COSYVOICE_LLM_KV_CACHE_TYPE_F16);
                    break;
                }
                else
                    cur_type = cosyvoice_llm_kv_cache_type_to_ggml(shared->params.llm_kv_cache_fallback);
            }
            else
                cur_type = cosyvoice_llm_kv_cache_type_to_ggml(shared->params.llm_kv_cache_type);

            if (shared->params.llm_kv_cache_separate_buffers)
            {
                auto v_type = ggml_is_quantized(cur_type) ? GGML_TYPE_F16 : cur_type;
                do
                {
                    if (attn_check(cur_type, v_type))
                    {
                        cv3_shared->k_type = cur_type;
                        cv3_shared->v_type = v_type;
                        break;
                    }

                    GGML_ASSERT(shared->params.llm_allow_kv_cache_fallback);
                    cur_type = cosyvoice_get_kv_fallback_type(cur_type);
                } while (cur_type != GGML_TYPE_F32);

                shared->params.llm_kv_cache_type = COSYVOICE_MAKE_SEPARATE_KV_CACHE(
                    cosyvoice_ggml_to_llm_kv_cache_type(cur_type),
                    cosyvoice_ggml_to_llm_kv_cache_type(v_type),
                    cosyvoice_ggml_to_llm_kv_cache_type(v_type));
            }
            else
            {
                if (attn_check(cur_type, cur_type))
                {
                    cv3_shared->k_type = cur_type;
                    cv3_shared->v_type = cur_type;
                    shared->params.llm_kv_cache_type = cosyvoice_ggml_to_llm_kv_cache_type(cur_type);
                    break;
                }
                else
                {
                    GGML_ASSERT(shared->params.llm_allow_kv_cache_fallback);
                    cur_type = GGML_TYPE_F16;
                }
                cv3_shared->k_type = cur_type;
                cv3_shared->v_type = cur_type;
                shared->params.llm_kv_cache_type = cosyvoice_ggml_to_llm_kv_cache_type(cur_type);
            }
        } while (false);
    }

    for (auto& worker : simple_span<cosyvoice_worker_context>(workers, shared->params.n_workers))
    {
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
    }

    if (shared->params.flow_use_flash_attn)
    {
        int heads = flow.decoder.estimator.transformer_blocks[0].attn.heads;
        auto q = ggml_new_tensor_4d(worker->ctx0.get(), GGML_TYPE_F32, flow.decoder.estimator.transformer_blocks[0].attn.to_q.weight->ne[1] / heads, 1, heads, 2);
        auto k = ggml_new_tensor_4d(worker->ctx0.get(), GGML_TYPE_F32, flow.decoder.estimator.transformer_blocks[0].attn.to_k.weight->ne[1] / heads, 1, heads, 2);
        auto v = ggml_new_tensor_4d(worker->ctx0.get(), GGML_TYPE_F32, flow.decoder.estimator.transformer_blocks[0].attn.to_v.weight->ne[1] / heads, 1, heads, 2);
        auto o = ggml_flash_attn_ext(worker->ctx0.get(), q, k, v, nullptr, 1.f / std::sqrt(static_cast<float>(k->ne[0])), 0.f, 0.f);
        shared->params.flow_use_flash_attn = ggml_backend_supports_op(backend, o);
    }

    {
        auto a = llm.embed_tokens_weight;
        auto b = ggml_cast(worker->ctx0.get(), a, GGML_TYPE_F32);
        shared->op_caps.emb_cast_f32 = ggml_backend_supports_op(backend, b);
    }

    for (auto& block : flow.decoder.estimator.transformer_blocks)
        block.attn.fattn = shared->params.flow_use_flash_attn;

    for (uint32_t i = 0; i < shared->params.n_workers; ++i)
    {
        auto cv3_worker = cv3_workers + i;
        auto worker = workers + i;

        switch (shared->params.inference_buffer_policy)
        {
        case COSYVOICE_INFERENCE_BUFFER_POLICY_BALANCED:
        case COSYVOICE_INFERENCE_BUFFER_POLICY_SHARED:
            cv3_worker->token2wav_buffer.reset(worker->kv_buffer.get());
        case COSYVOICE_INFERENCE_BUFFER_POLICY_DEDICATED:
            break;
        default:
            throw std::invalid_argument("unexpected policy");
        }
        cv3_worker->orig_max_seq_len = shared->params.n_max_seq;
    }

    auto arch = loader.get_string("general.architecture");
    shared->architecture.reset(new char[arch.size() + 1]);
    memcpy(shared->architecture.get(), arch.data(), arch.size() + 1);

    ggml_backend_synchronize(backend);
}

bool cosyvoice_model_3::set_worker_no(uint32_t worker_no)
{
    if (worker_no >= shared->params.n_workers)
        return false;

    worker = workers + worker_no;
    cv3_worker = cv3_workers + worker_no;
    return true;
}



void cosyvoice_model_3::empty_buffer_cache()
{
	cosyvoice_model::empty_buffer_cache();

    if (cv3_worker->orig_max_seq_len != shared->params.n_max_seq)
    {
        shared->params.n_max_seq = cv3_worker->orig_max_seq_len;
        worker->kv_buffer.reset(
            worker->kv_cache.initialize_buffer(
                worker->backend.get(),
                static_cast<int>(cv3_shared->llm.layers[0].self_attn.k_proj.weight->ne[1] / cv3_shared->llm.num_key_value_heads),
                static_cast<int>(cv3_shared->llm.layers[0].self_attn.v_proj.weight->ne[1] / cv3_shared->llm.num_key_value_heads),
                cv3_shared->llm.num_attention_heads,
                cv3_shared->llm.num_key_value_heads,
                shared->params.n_max_seq,
                cv3_shared->k_type,
                cv3_shared->v_type,
                shared->params.llm_use_flash_attn));

        switch (shared->params.inference_buffer_policy)
        {
        case COSYVOICE_INFERENCE_BUFFER_POLICY_BALANCED:
        case COSYVOICE_INFERENCE_BUFFER_POLICY_SHARED:
            cv3_worker->token2wav_buffer.release();
            cv3_worker->token2wav_buffer.reset(worker->kv_buffer.get());
            break;
        case COSYVOICE_INFERENCE_BUFFER_POLICY_DEDICATED:
            cv3_worker->token2wav_buffer.reset();
            break;
        default:
            GGML_ABORT("unexpected policy");
        }
    }
}

void cosyvoice_model_3::get_memory_usage(cosyvoice_memory_usage_t* usage)
{
    usage->parameters = ggml_backend_buffer_get_size(shared->buffer.get());
    usage->buffers = ggml_backend_sched_get_buffer_size(worker->sched.get(), worker->backend.get());
    usage->cpu_buffers = ggml_backend_sched_get_buffer_size(worker->sched.get(), worker->cpu_backend.get());
    usage->offloaded_kv_cache = worker->kv_cache.get_offloaded_cache_size();
    usage->random_noise = sizeof(float) * shared->rand_noise_len;
    usage->kv_cache = worker->kv_buffer.get() ? ggml_backend_buffer_get_size(worker->kv_buffer.get()) : 0;
    usage->token2wav = cv3_worker->token2wav_buffer.get() ? ggml_backend_buffer_get_size(cv3_worker->token2wav_buffer.get()) : 0;
}

void cosyvoice_model_3::get_total_memory_usage(cosyvoice_memory_usage_t* usage)
{
    usage->parameters = ggml_backend_buffer_get_size(shared->buffer.get());
    for (uint32_t i = 0; i != shared->params.n_workers; ++i)
    {
        auto worker = workers + i;
        auto cv3_worker = cv3_workers + i;

        usage->buffers = ggml_backend_sched_get_buffer_size(worker->sched.get(), worker->backend.get());
        usage->cpu_buffers = ggml_backend_sched_get_buffer_size(worker->sched.get(), worker->cpu_backend.get());
        usage->offloaded_kv_cache = worker->kv_cache.get_offloaded_cache_size();
        usage->random_noise = sizeof(float) * shared->rand_noise_len;
        usage->kv_cache = worker->kv_buffer.get() ? ggml_backend_buffer_get_size(worker->kv_buffer.get()) : 0;
        usage->token2wav = cv3_worker->token2wav_buffer.get() ? ggml_backend_buffer_get_size(cv3_worker->token2wav_buffer.get()) : 0;
    }
}

void cosyvoice_model_3::reset_shared_buffer(ggml_backend_buffer* new_buffer)
{
    cv3_worker->token2wav_buffer.reset(new_buffer);
    if (shared->params.inference_buffer_policy != COSYVOICE_INFERENCE_BUFFER_POLICY_DEDICATED)
    {
        worker->kv_buffer.release();
        worker->kv_buffer.reset(new_buffer);
        shared->params.n_max_seq = worker->kv_cache.reset_buffer(new_buffer);
    }
}

cosyvoice_3_worker_context::cosyvoice_3_worker_context() :
    ctx1(ggml_init(ggml_init_params{ ggml_tensor_overhead() * 4, nullptr, true })) {}

cosyvoice_model_3::cosyvoice_model_3(ggml_backend_t backend, const cosyvoice_context_params_v2_cpp& params)
    : cosyvoice_model(backend, params), cv3_shared(new cosyvoice_model_3_shared), cv3_workers(new cosyvoice_3_worker_context[params.n_workers]())
{
    cv3_worker = cv3_workers;
}

cosyvoice_model_3::~cosyvoice_model_3()
{
    if (get_ref_count() == 1)
    {
        delete cv3_shared;
        delete[] cv3_workers;
    }
}

void CausalHiFTGenerator::set_rand_ini(const float* data) const
{
    ggml_backend_tensor_set(m_source.l_sin_gen.rand_ini, data, 0, (nb_harmonics + 1) * sizeof(float));
}

bool cosyvoice_model_3::llm_is_stop_token(int token_id)
{
    return cv3_shared->stop_tokens.find(token_id) != cv3_shared->stop_tokens.end();
}

const ggml_tensor* cosyvoice_model_3::get_word_token_embed_weight()
{
    return cv3_shared->llm.embed_tokens_weight;
}

const ggml_tensor* cosyvoice_model_3::get_speech_token_embed_weight()
{
    return cv3_shared->llm.speech_embedding_weight;
}

uint32_t cosyvoice_model_3::get_hift_rand_ini_len()
{
    return cv3_shared->hift.nb_harmonics + 1;
}

void cosyvoice_model_3::set_hift_rand_ini(const float* data)
{
    cv3_shared->hift.set_rand_ini(data);
}

uint32_t cosyvoice_model_3::get_sample_rate()
{
    return cv3_shared->hift.sampling_rate;
}

void cosyvoice_model::get_default_generation_config(cosyvoice_generation_config_t* config)
{
    *config = shared->config;
}

void cosyvoice_model::get_generation_config(cosyvoice_generation_config_t* config)
{
    *config = worker->config;
}

bool cosyvoice_model::set_generation_config(const cosyvoice_generation_config_t* config)
{
	if (config->max_token_text_ratio < config->min_token_text_ratio
		|| config->min_token_text_ratio < 0.f)
		return false;

	if (config->temperature <= 0.f)
		return false;

	if (config->sampling.win_size <= 0
		|| config->sampling.top_k < 0
		|| config->sampling.top_p < 0.f
		|| config->sampling.top_p > 1.f
		|| config->sampling.tau_r < 0.f)
		return false;

    worker->config = *config;
    auto required_capacity = static_cast<uint32_t>(config->sampling.top_k * 2);
    if (worker->nucleus_probs_capacity < required_capacity)
    {
        worker->nucleus_probs.reset(new float[required_capacity]);
        worker->nucleus_probs_capacity = required_capacity;
    }
    worker->nucleus_probs_len = 0;
    return true;
}

const char* cosyvoice_model::get_instruction_prefix()
{
    return shared->instruction_prefix.get();
}

void cosyvoice_model::get_context_params(cosyvoice_context_params_t* params)
{
    *params = shared->params;
    params->seed = worker->sampler_seed;
    params->builtin_sampler_rng_policy = worker->builtin_sampler_rng_policy;
    get_sampler(&params->sampler, &params->sampler_ctx);
}

const char* cosyvoice_model::get_architecture()
{
    return shared->architecture.get();
}

bool cosyvoice_model::is_backend_uma()
{
    return shared->backend_uma;
}

void cosyvoice_model::get_sampler(cosyvoice_sampler_t* sampler, void** sampler_ctx)
{
    if (using_builtin_sampler())
    {
        *sampler = nullptr;
        *sampler_ctx = nullptr;
    }
    else
    {
        *sampler = shared->params.sampler;
        *sampler_ctx = shared->params.sampler_ctx;
    }
}

void cosyvoice_model::set_sampler(cosyvoice_sampler_t sampler, void* sampler_ctx)
{
    if (sampler)
    {
        shared->params.sampler = sampler;
        shared->params.sampler_ctx = sampler_ctx;
    }
    else
    {
        shared->params.sampler = reinterpret_cast<cosyvoice_sampler_t>(cosyvoice_llm_sampler);
        shared->params.sampler_ctx = workers;
        reset_builtin_sampler_rng();
    }
}

void cosyvoice_model::set_noise_callback(cosyvoice_noise_callback_t callback, void* callback_ctx)
{
    if (callback)
    {
        std::unique_lock lock(shared->noise_mutex);
        shared->noise_callback = callback;
        shared->noise_callback_ctx = callback_ctx;
    }
    else
    {
        std::unique_lock lock(shared->noise_mutex);
        shared->noise_callback = reinterpret_cast<cosyvoice_noise_callback_t>(cosyvoice_default_noise_callback);
        shared->noise_callback_ctx = shared;
    }
}

void cosyvoice_model::get_noise_callback(cosyvoice_noise_callback_t* callback, void** callback_ctx)
{
    std::shared_lock lock(shared->noise_mutex);
    if (shared->noise_callback == reinterpret_cast<cosyvoice_noise_callback_t>(cosyvoice_default_noise_callback))
    {
        *callback = nullptr;
        *callback_ctx = nullptr;
    }
    else
    {
        *callback = shared->noise_callback;
        *callback_ctx = shared->noise_callback_ctx;
    }
}


int cosyvoice_model::llm_sample_token()
{
    GGML_ASSERT(worker->llm_probs);
    return shared->params.sampler_ext(
        reinterpret_cast<cosyvoice_llm_token_prob_t*>(worker->nucleus_probs.get()),
        worker->nucleus_probs_len,
        worker->probs.get(),
        static_cast<uint32_t>(get_speech_token_embed_weight()->ne[1]),
        &worker->config.sampling,
        worker->tokens.data(),
        static_cast<uint32_t>(worker->tokens.size()),
        shared->params.sampler_ctx,
        static_cast<uint32_t>(worker - workers));
}

void cosyvoice_model::llm_accept_token(int token)
{
    worker->tokens.push_back(token);
}

void cosyvoice_model::llm_clear_accepted_tokens()
{
    worker->tokens.clear();
}

uint32_t cosyvoice_model::llm_get_n_accepted_tokens()
{
    return static_cast<uint32_t>(worker->tokens.size());
}

const int* cosyvoice_model::llm_get_accepted_tokens()
{
    return worker->tokens.data();
}

ggml_status cosyvoice_model::get_last_status()
{
    return worker->status;
}

uint32_t cosyvoice_model::get_worker_no()
{
    return static_cast<uint32_t>(worker - workers);
}

uint32_t cosyvoice_model::get_n_workers()
{
    return shared->params.n_workers;
}

uint32_t cosyvoice_model::llm_get_kv_cache_len()
{
    return worker->kv_cache.cur_len;
}

bool cosyvoice_model::llm_set_kv_cache_len(uint32_t len)
{
    auto& cur_len = worker->kv_cache.cur_len;
    if (len <= cur_len)
    {
        cur_len = len;
        return true;
    }
    return false;
}
