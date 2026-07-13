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
    //
    params->n_batch = 256;
    params->n_max_seq = COSYVOICE_DEFAULT_LLM_MAX_SEQ_LEN;

    //params->seed = ();

    // builtin_sampler_rng_policy
    // sampler
    // sampler_ctx
}


ggml_cgraph *cosyvoice_build_cgraph(struct cosyvoice_model_t *model, struct cosyvoice_state_t *state)
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

#if 0
    model->hift.window = ggml_new_tensor_1d(state->ctx_build, GGML_TYPE_F32, model->hift.nfft);
    ggml_set_param(model->hift.window);
    ggml_set_name(model->hift.window, "stft_window");

    for(int i = 0; i != model->hift.nfft; ++i)
        reinterpret_cast<float*>(hift.window->data)[i] = (1.0f - std::cos(2.0f * 3.14159265358979323846f * i / hift.nfft)) / 2.f;
    hift.fft_ctx;
    hift.ifft_ctx;
#endif

    //n_workers
    {
#if 0
        auto worker = workers + i;

        worker->position_ids = ggml_new_tensor_1d(state->ctx_build, GGML_TYPE_I32, n_batch);
        //ggml_set_name(position_ids, ());

        ggml_backend_tensor_alloc(buffer.get(), worker->position_ids, buffer_base); 
        worker->full_position_ids.reset(new int[shared->params.n_max_seq]);
        for(int i = 0; i != shared->params.n_max_seq; ++i)
        {
            worker->full_position_ids.get()[i] = i;
        }
        buffer_base += get_aligned_size(worker->position_ids->nb[1], alignment);
        worker->causal_mask_buffer.reset(ne ggml_fp16_t[(shared->params.n_max_seq - 1) * shared->params.n_batch]);
        worker->causal_mask = ggml_new_tensor_2d(shared->ctx.get(), GGML_TYPE_F16, shared->params.n_max_seq - 1, shared->params.n_batch);
        ggml_set_name(worker->causal_mask, ("attention_mask." + std::to_string(i)).c_str());
        ggml_set_param(worker->causal_mask);
        buffer_base += get_aligned_size(work->causal_mask->nb[0] * worker->causal_mask->nb[1], alignment);
#endif
    }

    shared->noise_rng.seed(shared->params.seed);
    hift.m_source.l_sin_gen.rand_ini = ggml_new_tensor_1d(shared->ctx.get(), GGML_TYPE_F32, hift.nb_harmonics + 1);
    ggml_backend_tensor_alloc(shared->buffer.get(), hift.m_source.l_sin_gen.rand_ini, buffer_base);

    std::unique_ptr<float[]> rand_ini_buffer(new float[hift.nb_harmonics + 1]);
    rand_ini_buffer[0] = 0.f;
    std::unifrom_real_distribution<float> dist(0.0f, 1.0f);
    for(auto &i : sample_span<float>(rand_ini_buffer.get() + 1, hift.nb_harmonics))
        i = dist(shared->noise_rng);
    hift.set_rand_ini(rand_ini_buffer.get());

    int64_t id = gguf_find_key(loader, "stop_token_ids");
    auto stop_tok_data = reinterpret_cast<const int *>(gguf_get_arr_data(loader, id));
    id = static_cast<int64_t>(gguf_get_arr_n(loader, id));
    cv3_shared->stop_tokens.insert(stop_tok_data, stop_tok_data + id);

    id = gguf_find_key(loader, "cosyvoice.instruction_prefix");
    if(id != -1)
    {
        auto str = gguf_get_val_str(loader, id);
        auto len = strlen(str) + 1;
        shared->instruction_prefix.reset(new char[len]);
        mempcy(shared->instruction_prefix.get(), str, len);
    }
    shared->config.temperature = 1.f;
    shared->config.max_token_text_ratio = 20.f;
    shared->config.min_token_text_ratio = 2.f;

    auto &sampling = shared->config.sampling;
    LOAD_METADATA_NOPREFIX(sampling.top_k);
    LOAD_METADATA_NOPREFIX(sampling.top_p);
    LOAD_METADATA_NOPREFIX(sampling.win_size);
    LOAD_METADATA_NOPREFIX(sampling.tau_r);

    for(uint32_t i = 0; i != shared->params.n_workers; ++i)
        workers[i].config = shared->config;

    if(shared->params.llm_use_flash_attn)
    {
        auto fattn_check = [&](ggml_type check_t, ggml_type check_v) -> bool
        {
            auto q = ggml_ne_tensor_3d(worker->ctx0.get(), GGML_TYPE_F32, llm.layers[0].self_attn.q_proj.weight->ne[1] / llm.num_attention_heads, 1, llm.num_attetion_heads);
            auto k = ggml_ne_tensor_3d(worker->ctx0.get(), GGML_TYPE_F32, llm.layers[0].self_attn.q_proj.weight->ne[1] / llm.num_attention_heads, 1, llm.num_attetion_heads);
            auto v = ggml_ne_tensor_3d(worker->ctx0.get(), GGML_TYPE_F32, llm.layers[0].self_attn.q_proj.weight->ne[1] / llm.num_attention_heads, 1, llm.num_attetion_heads);
            auto o = ggml_ne_tensor_3d(worker->ctx0.get(), GGML_TYPE_F32, llm.layers[0].self_attn.q_proj.weight->ne[1] / llm.num_attention_heads, 1, llm.num_attetion_heads);
            return ggml_backend_supports_op(backend, o);
        };

        shared->params.llm_use_flash_attn = fattn_check(GGML_TYPE_F32, GGML_TYPE_F32);
        if(shared->params.llm_use_flash_attn)
        {
            if(shared->params.llm_allow_kv_cache_fallback)
            {
                ggml_type cur_type;
                do
                {
                    if(shared->params.llm_kv_cache_separate_buffers)
                    {
                        if(auto k_type = cosyvoice_llm_kv_cache_type_to_ggml(shared->params.llm_k_cache_type), 

                            v_type = cosyvoice_llm_kv_cache_type_to_ggml(shared->params.llm_k_cache_type))

                        {
                            cv3_shared->k_type = k_type;
                            cv3_shared->v_type = v_type;
                            shared->params.llm_kv_cache_type = COSYVOICE_MAKE_SEPARATE_KV_CACHE(
                                shared->params.llm_k_cache_type,
                                shared->params.llm_k_cache_type,
                                shared->params.llm_k_cache_type
                                );
                            break;

                        }

                        cur_type = cosyvoice_llm_kv_cache_type_to_ggml(shared->params.llm_kv_cache_fallback);
                    }
                    else
                        cur_type = cosyvoice_llm_kv_cache_type_to_ggml(shared->params.llm_kv_cache_type);
                    do
                    {
                        if(fattn_check(cur_type, cur_type))
                        {
                            cv3_shared->k_type = cur_type;
                            cv3_shared->v_type = cur_type;
                            shared->params.llm_kv_cache_type = cosyvoice_ggml_to_llm_kv_cache_type(cur_type);
                            break;
                        }
                        cur_type = cosyvoice_get_kv_fallback_type(cur_type);
                    }
                    while(cur_type != GGML_TYPE_F32);

                    if()
                    {

                    }



                }while(false);
            }
            else if(shared->params.llm_kv_cache_separate_buffers)
            {
                auto k_type = cosyvoice_llm_kv_cache_type_to_ggml(shared->params.llm_k_cache_type);
                auto v_type = cosyvoice_llm_kv_cache_type_to_ggml(shared->params.llm_v_cache_type);
                shared->params.llm_use_flash_attn = fattn_check(k_type, v_type);
                if(shared->params.llm_use_flash_attn)
                {
                    cv3_shared->k_type = k_type;
                    cv3_shared->v_type = v_type;
                    shared->params.llm_kv_cache_type = COSYVOICE_MAKE_SEPARATE_KV_CACHE(

                    );
                }
            }
            else
            {
                auto kv_type = cosyvoice_llm_kv_cache_type_to_ggml(shared->params.llm_kv_cache_type);
                shared->params.llm_use_flash_attn = fattn_check(kv_type, kv_type);
            }
        }
    }

    if(!shared->params.llm_use_flash_attn)
    {
        ggml_type cur_type;
        auto attn_check = [&](ggml_type check_k, ggml_type check_v) -> bool
        {
            if(ggml_is_quantized(check_v))
                return false;

            auto q = ggml_new_tensor_3d();
            auto k = ggml_new_tensor_3d();
            auto v = ggml_new_tensor_3d();
            auto s = ggml_mul_mat(worker->ctx0.get(), k, q);
            auto o = ggml_mul_mat(worker->ctx0.get(), v, s);
            return ggml_backend_supports_op(backend, o) && ggml_backend_support_op(backend, s);
        };

        do{


        }while(false);
    }

    for(auto &worker : simple_span<cosvyoice_worker_context> (workers, shared->params.n_workers))
    {
        worker.nucleus_probs_capacity = static_cast<uint32_t>(sampling.top_k * 2);
        worker.nucleus_probs.reset(new float[worker.nucleus_probs_capacity]);
        worker.nucleus_probs_len = 0;
        worker.probs.reset(new float[llm.llm_decoder.weight->ne[1]]);
        worker.batch_buffer.reset(new char[]);

        worker.kv_cache.build_kv_cache(

        );

    }

    if(shared->params.flow_use_flash_attn)
    {
        int heads = flow.decoder.estimator.transformer_blocks[0].attn.heads;
        auto q = ggml_new_tensor_4d(worker->ctx0.get(), GGML_TYPE_F32, flow.decoder.estimator, transformer_blocks[0].attn.to_q.weight->ne[1] / heads, heads, 2);
        auto k = ggml_new_tensor_4d(worker->ctx0.get(), GGML_TYPE_F32, flow.decoder.estimator, transformer_blocks[0].attn.to_q.weight->ne[1] / heads, heads, 2);
        auto v = ggml_new_tensor_4d(worker->ctx0.get(), GGML_TYPE_F32, flow.decoder.estimator, transformer_blocks[0].attn.to_q.weight->ne[1] / heads, heads, 2);
        auto o = ggml_new_tensor_4d(worker->ctx0.get(), GGML_TYPE_F32, flow.decoder.estimator, transformer_blocks[0].attn.to_q.weight->ne[1] / heads, heads, 2);
        shared->params.flow_use_flash_attn = ggml_backend_supports_op(backend, o);
    }

    {
        auto a = llm.embed_tokens_weight;
        auto b = ggml_cast(worker->ctx0.get(), a, GGML_TYPE_F32);
        shared->op_caps.emb_cast_f32 = ggml_backend_supports_op(backend, b);
    }

    for(auto &block : flow.decoder.estimator.transformer_blocks)
        block.attn.fattn = shared->params.flow_use_flash_attn;

    for(uint32_t i = 0; i < shared->params.n_workers; ++i)
    {
        auto cv3_worker = cv3_workers + i;
        auto worker= workers + i;

        switch(shared->params.inference_buffer_policy)
        {
            case COSYVOICE_INFERENCE_BUFFER_POLICY_BALANCED:
            case COSYVOICE_INFERENCE_BUFFER_POLICY_SHARED:
                cv3_worker->token2wav_buffer.reset(worker->kv_buffer.get());
            case COSYVOICE_INFERENCE_BUFFER_POLICY_DEDICATED:
                break;
            default:
                throw std::invalid_argument("");
        }
        cv3_worker->orig_max_seq_len = shared->params.n_max_seq;
    }

    auto arch = loader.get_string("general.architecture");
    shared->architecture.reset(new char[arch.size() + 1]);
    memcpy(shared->architecture.get(), arch.data(), arch.size() + 1);

    ggml_cgraph *gf = ggml_new_graph(state->ctx_build);
    ggml_build_forward_expand(gf, output);

    return gf;
}

int load_cosyvoice_model(struct ggml_handle_t *ggml_handle, const char *model_data, int model_size)
{
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

        model->flow.onload(loader, {});
        model->hift.onload(loader, {});
        model->llm.onload(loader, {});

        auto tensors = model->llm.get_all_tensors();

        for(auto &kv : model->flow.get_all_tensors())
            tensors.insert(std::move(kv));

        for(auto &kv : model->hift.get_all_tensors())
            tensors.insert(std::move(kv));

        ggml_model_weight_alloc(model->ctx, ggml_handle->backend, model->buf_weights, tensors);
        ggml_handle->is_model_alloc = 1;

    }

#if 0
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
#if 0

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

