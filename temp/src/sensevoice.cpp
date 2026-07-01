#include "base.h"
#include "sensevoice.h"
#include "gguf_loader.h"
#include <ggml-backend.h>
#include <thread>

#define WARP_SIZE 32

inline static const char *get_tensor_type_string(int type)
{
    switch (type)
    {
        case GGML_TYPE_F16:
            return "FP16";
        case GGML_TYPE_F32:
            return "FP32";
        case GGML_TYPE_I8:
            return "INT8";
        case GGML_TYPE_I16:
            return "INT16";
        case GGML_TYPE_I32:
            return "INT32";
        case GGML_TYPE_I64:
            return "INT64";
        default:
            return "Q_K";
    }
}


void print_tensor_shape(struct ggml_tensor *tensor)
{
    int i;
    printf("%s shape  [", tensor->name);
    for(i = 0; i < ggml_n_dims(tensor); i++)
    {
        printf("%d ", tensor->ne[i]);
    }
    printf("] ");
    printf(" TENSOR DATA TYPE %s \n", get_tensor_type_string(tensor->type));
}



// faster matrix multiplications for tensors that do not have dimension 0 divisible by "pad"
// the idea is to represent the original matrix multiplication:
//
//   Z = X @ Y
//
// with the sum of two matrix multiplications:
//
//   Z = (X_0 @ Y_0) + (X_1 @ Y_1)
//
// here X_0 and Y_0 are views of X and Y that have dimension 0 divisible by "pad"
// and X_1 and Y_1 are the remaining views. X_1 and Y_1 end up being small matrices that can be processed with more
// general-purpose kernels
//
static struct ggml_tensor * ggml_mul_mat_pad(struct ggml_context * ctx, struct ggml_tensor * x, struct ggml_tensor * y, int pad = 32) {
    // use padding only if dimension 0 is at least 8 times larger than the padding
    // else we won't get much benefit from the optimization
    const int n_pad_req = 8;

    if (x->ne[0] % pad == 0 || x->ne[0] / pad < n_pad_req) {
        return ggml_mul_mat(ctx, x, y);
    }

    struct ggml_tensor * x_0 = ggml_view_3d(ctx, x, (x->ne[0]/pad)*pad, x->ne[1], x->ne[2], x->nb[1], x->nb[2], 0);
    struct ggml_tensor * x_1 = ggml_view_3d(ctx, x,  x->ne[0]%pad,      x->ne[1], x->ne[2], x->nb[1], x->nb[2], x_0->ne[0]*x_0->nb[0]);

    struct ggml_tensor * y_0 = ggml_view_3d(ctx, y, (y->ne[0]/pad)*pad, y->ne[1], y->ne[2], y->nb[1], y->nb[2], 0);
    struct ggml_tensor * y_1 = ggml_view_3d(ctx, y,  y->ne[0]%pad,      y->ne[1], y->ne[2], y->nb[1], y->nb[2], y_0->ne[0]*y_0->nb[0]);

    return ggml_add(ctx,
                    ggml_mul_mat(ctx, x_0, y_0),
                    ggml_mul_mat(ctx, x_1, y_1));
}

// copy from whisper.cpp
// TODO: CUDA is currently broken - seems ggml_mul_mat does not handle views correctly
#if defined(GGML_USE_METAL)
#define ggml_mul_mat ggml_mul_mat_pad
#endif


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

void sensevoice_default_params(sensevoice_context_params_t *params)
{
    params->inference_buffer_policy = NN_INFERENCE_BUFFER_POLICY_BALANCED;
    params->n_batch = 256; //to do 

#if 0
    std::random_device rd;
    if (rd.entropy() == 0)
        params->seed = static_cast<uint32_t>(std::chrono::system_clock::now().time_since_epoch().count());
    else
        params->seed = rd();
    params->builtin_sampler_rng_policy = COSYVOICE_BUILTIN_SAMPLER_RNG_POLICY_RESET_PER_SESSION;
#endif
    //sampler 
    //sampler_ctx
}

void init_sensevoice_backend()
{
    ggml_time_init();
    ggml_log_set(nn_log_callback_default, nullptr);

    ggml_init_params params{};
    ggml_context *ctx = ggml_init(params);
    ggml_free(ctx);
}


sensevoice_context_t load_sensevoice_model(const char *filename,
                                const sensevoice_context_params_t params,
                                ggml_backend_t backend,
                                uint32_t n_threads,
                                uint32_t reserved)
{

    gguf_loader loader(filename);
    if(!loader)
        return nullptr;

    auto ctx = new sensevoice_context(params, backend ? backend : ggml_backend_init_best());
    ctx->sensevoice_model::load(loader);

    auto ggml_backend_set_n_threads = reinterpret_cast<ggml_backend_set_n_threads_t>(ggml_backend_reg_get_proc_address(ggml_backend_dev_backend_reg(ggml_backend_get_device(ctx->cpu_backend.get())), "ggml_backend_set_n_threads"));

    if (n_threads == 0)
        n_threads = std::thread::hardware_concurrency();
    if (n_threads != 0)
        ggml_backend_set_n_threads(ctx->cpu_backend.get(), n_threads);

    ctx->sensevoice_model::build_cgraph();

    ctx->sensevoice_model::inference();
    return ctx;
}

sensevoice_model::~sensevoice_model()
{
    //to do 
#if 0
    delete kv_cache;

    if(backend == cpu_backend)
        cpu_backend.release();

    switch(params.inference_buffer_policy)
    {
        case NN_INFERENCE_BUFFER_POLICY_BALANCED:
        case NN_INFERENCE_BUFFER_POLICY_SHARED:
            kv_buffer.release();
            break;
    }
#endif
}

void sensevoice_model::empty_buffer_cache()
{
    ggml_backend_t backends[] = { backend.get(), cpu_backend.get() };
    sched.reset(ggml_backend_sched_new(
        backends,
        nullptr,
        backend == cpu_backend ? 1 : 2,
        GGML_DEFAULT_GRAPH_SIZE * 4,
        true,
        true
    )); 

    //if (kv_cache)
    //    kv_cache->clear_offloaded_cache();
}

sensevoice_model::sensevoice_model(ggml_backend_t backend, const sensevoice_context_params_t &params) :  backend(backend), params(params), cpu_backend(backend),
    ctx0(ggml_init(ggml_init_params{ .mem_size = ggml_graph_overhead() * GGML_DEFAULT_GRAPH_SIZE, .no_alloc = true })),
    gf(nullptr), 
    status(GGML_STATUS_SUCCESS)
{
    if (GGML_BACKEND_DEVICE_TYPE_CPU != ggml_backend_dev_type(ggml_backend_get_device(backend)))
    {
        cpu_backend.release();
        cpu_backend.reset(ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr));
    }

    empty_buffer_cache();

    ggml_tensor* a = ggml_new_tensor_1d(ctx0.get(), GGML_TYPE_I32, 1);
    a = ggml_concat(ctx0.get(), a, a, 0);
    op_caps.concat_i32 = ggml_backend_supports_op(backend, a);

    a = ggml_new_tensor_3d(ctx0.get(), GGML_TYPE_F16, 11, 4, 51);
    a = ggml_repeat_4d(ctx0.get(), a, 11, 4, 51, 4);
    op_caps.repeat_f16 = ggml_backend_supports_op(backend, a);
}


void sensevoice_embedding::OnLoad(const gguf_loader &loader, const std::string &prefix)
{
    LOAD_TENSOR_EX("embed.weight", weight);
}

void sensevoice_layer_encoder::OnLoad(const gguf_loader &loader, const std::string &prefix)
{
    LOAD_TENSOR_EX("self_attn.linear_out.weight",  e_attn_ln_out_w);
    LOAD_TENSOR_EX("self_attn.linear_out.bias",  e_attn_ln_out_b);

    LOAD_TENSOR_EX("self_attn.linear_q.weight",  e_attn_ln_q_w);
    LOAD_TENSOR_EX("self_attn.linear_q.bias",  e_attn_ln_q_b);

    LOAD_TENSOR_EX("self_attn.linear_k.weight",  e_attn_ln_k_w);
    LOAD_TENSOR_EX("self_attn.linear_k.bias",  e_attn_ln_k_b);

    LOAD_TENSOR_EX("self_attn.linear_v.weight",  e_attn_ln_v_w);
    LOAD_TENSOR_EX("self_attn.linear_v.bias",  e_attn_ln_v_b);

    LOAD_TENSOR_EX("self_attn.fsmn_block.weight",  e_attn_fsmn_w);

    LOAD_TENSOR_EX("feed_forward.w_1.weight",  e_mlp_w1);
    LOAD_TENSOR_EX("feed_forward.w_1.bias",  e_mlp_b1);

    LOAD_TENSOR_EX("feed_forward.w_2.weight",  e_mlp_w2);
    LOAD_TENSOR_EX("feed_forward.w_2.bias",  e_mlp_b2);

    LOAD_TENSOR_EX("norm1.weight",  e_norm_w1);
    LOAD_TENSOR_EX("norm1.bias",  e_norm_b1);

    LOAD_TENSOR_EX("norm2.weight",  e_norm_w2);
    LOAD_TENSOR_EX("norm2.bias",  e_norm_b2);
}

ggml_tensor *sensevoice_layer_encoder::build_cgraph(ggml_context* ctx0, ggml_cgraph* gf, ggml_tensor* cur, sensevoice_context_params_t &params, sensevoice_kv_cache &kv_cache, bool user_flash_attn) const 
{
    const int n_state = params.n_encoder_hidden_state;
    const int n_head = params.n_encoder_attention_heads;

    ggml_tensor *residual = nullptr;

    if (e_norm_w1->ne[0] == e_norm_w2->ne[0]) {
        residual = ggml_cpy(
                ctx0, cur,
                ggml_new_tensor_3d(ctx0, cur->type, cur->ne[0], cur->ne[1], cur->ne[2]));
    }

    {
        // layer norm
        // cur = ln_0_w*cur + ln_0_b
#ifdef GGML_CUDA
        int32_t dim_size = cur->ne[0];
        if (sctx.params.use_gpu && dim_size % WARP_SIZE) {
            int32_t pad_size = WARP_SIZE - (dim_size % WARP_SIZE);
            ggml_tensor *mean = ggml_mean(ctx0, cur);
            cur = ggml_sub(ctx0, cur, mean);
            ggml_tensor *sigma = ggml_mul(ctx0, cur, cur);
            sigma = ggml_sum_rows(ctx0, sigma);
            cur = ggml_scale(ctx0, ggml_div(ctx0, cur, ggml_sqrt(ctx0, sigma)), sqrt(dim_size));
            // cur = ggml_cont(ctx0, ggml_pad(ctx0, cur, pad_size, 0, 0, 0));
            // cur = ggml_norm(ctx0, cur, hparams.eps);
            // cur = ggml_cont(ctx0, ggml_view_4d(ctx0, cur, dim_size, cur->ne[1], cur->ne[2], cur->ne[3], cur->nb[1], cur->nb[2], cur->nb[3], 0));
            // cur = ggml_scale(ctx0, cur, sqrt(float(dim_size) / (dim_size + pad_size)));
        }else{
            cur = ggml_norm(ctx0, cur, params.eps);
        }
#else
        cur = ggml_norm(ctx0, cur, params.eps);
#endif
        cur = ggml_add(ctx0, ggml_mul(ctx0, cur, e_norm_w1), e_norm_b1);
    }

    // self attention
    {
        // self attention linear qkv
        //      cur = ggml_transpose(ctx0, cur);
        // split qkv into separate tensors
        // q, k, v = torch.split(q_k_v, int(self.h * self.d_k), dim=-1)
        //  ref:
        //  https://github.com/alibaba-damo-academy/FunASR/blob/main/funasr/modules/attention.py#L391-L396
        struct ggml_tensor *Q;
        struct ggml_tensor *Q_h;
        struct ggml_tensor *K;
        struct ggml_tensor *K_h;
        struct ggml_tensor *V;
        struct ggml_tensor *V_h;

        int n_ctx = cur->ne[1];
        int n_batch = cur->ne[2];

        ggml_set_name(e_attn_ln_q_w, "e_attn_ln_q_w");
        //print_tensor_shape(e_attn_ln_q_w);
        ggml_set_name(cur, "cur");
        //print_tensor_shape(cur);


        Q = ggml_add(ctx0,
                     ggml_mul_mat_pad(ctx0, e_attn_ln_q_w, cur),
                     e_attn_ln_q_b);

        Q_h = ggml_reshape_4d(ctx0, Q, n_state / n_head, n_head, n_ctx, n_batch);
        Q_h = ggml_permute(ctx0, Q_h,  0, 2, 1, 3);
        Q_h = ggml_cont(ctx0, Q_h);

        ggml_set_name(Q_h, "attention_Q");

        K = ggml_add(ctx0,
                     ggml_mul_mat(ctx0, e_attn_ln_k_w, cur),
                     e_attn_ln_k_b);

        K_h = ggml_reshape_4d(ctx0, K, n_state / n_head, n_head, n_ctx, n_batch);
        K_h = ggml_permute(ctx0, K_h, 0, 2, 1, 3);
        K_h = ggml_cont(ctx0, K_h);
        //      K = ggml_reshape_3d(ctx0, K, n_state, n_ctx, n_head);
        ggml_set_name(K_h, "attention_K");

        V = ggml_add(ctx0,
                     ggml_mul_mat(ctx0, e_attn_ln_v_w, cur),
                     e_attn_ln_v_b);
        ggml_set_name(V, "attention_V");

        V_h = ggml_reshape_4d(ctx0, V, n_state / n_head, n_head, n_ctx, n_batch);
        V_h = ggml_permute(ctx0, V_h, 0, 2, 1, 3);
        V_h = ggml_cont(ctx0, V_h);

        // fsmn forward with V
        int padding = (params.fsmn_kernel_size - 1) / 2;

        struct ggml_tensor *fsmn_memory = nullptr;
        // conv depth wise
        {
            {
                // implement conv depth wise with groups=input_channel implement
                // same in pytorch : F.conv1d(input, weight, bias=None, stride=1, padding=1, dilation=1, groups=n_state)
                struct ggml_tensor * a = e_attn_fsmn_w;
                struct ggml_tensor * b = ggml_cont(ctx0, ggml_transpose(ctx0, V));
                // Process each batch separately and concatenate results
                // for (int i = 0; i < b->ne[2]; i++) {
                //     // View for current batch
                //     struct ggml_tensor *b_batch = ggml_view_3d(ctx0, b, b->ne[0], b->ne[1], 1, b->nb[1], b->nb[2], i * b->nb[2]);
                //     struct ggml_tensor *im2col = ggml_im2col(ctx0, a, ggml_reshape_4d(ctx0, b_batch, b_batch->ne[0], 1, b_batch->ne[1], b_batch->ne[2] * b_batch->ne[3]), 1, 0, padding, 0, 1, 0, false, GGML_TYPE_F32);
                //     struct ggml_tensor * result = ggml_mul_mat(ctx0, a, im2col);
                //     struct ggml_tensor * fsmn_memory_batch = ggml_reshape_3d(ctx0, result, im2col->ne[1], b_batch->ne[1], b_batch->ne[2]);
                //     if (fsmn_memory == nullptr) {
                //         fsmn_memory = fsmn_memory_batch;
                //     } else {
                //         fsmn_memory = ggml_concat(ctx0, fsmn_memory, fsmn_memory_batch, 2);
                //     }
                // }
                struct ggml_tensor * im2col = ggml_im2col(ctx0, a, ggml_reshape_4d(ctx0, b, b->ne[0], 1, b->ne[1] * b->ne[2], b->ne[3]), 1, 0, padding, 0, 1, 0, false, GGML_TYPE_F32);
                im2col = ggml_reshape_4d(ctx0, im2col, im2col->ne[0], im2col->ne[1], im2col->ne[2] / n_batch, n_batch);
                a = ggml_repeat(ctx0, ggml_cast(ctx0, a, GGML_TYPE_F32), ggml_new_tensor_4d(ctx0, GGML_TYPE_F16, a->ne[0], a->ne[1], a->ne[2], n_batch));
                struct ggml_tensor * result = ggml_mul_mat(ctx0, a, im2col);
                fsmn_memory = ggml_reshape_3d(ctx0, result, im2col->ne[1], im2col->ne[2], im2col->ne[3]);
            }
            fsmn_memory = ggml_cont(ctx0, ggml_transpose(ctx0, fsmn_memory));
            fsmn_memory = ggml_add(ctx0, fsmn_memory, V);
            ggml_set_name(fsmn_memory, "fsmn_memory");
        }

        float KQscale = 1.0f / sqrtf(float(n_state) / n_head);


        if(user_flash_attn){
            const int n_ctx_pad = GGML_PAD(n_ctx, 256);
            const int n_state_head = n_state / n_head;

            //ggml_build_forward_expand(gf, ggml_cpy(ctx0, K, ggml_view_1d(ctx0, kv_cache.k, n_ctx*n_state*n_batch, 0)));
            //ggml_build_forward_expand(gf, ggml_cpy(ctx0, V, ggml_view_1d(ctx0, kv_cache.v, n_ctx*n_state*n_batch, 0)));

            struct ggml_tensor * K =
                    ggml_view_4d(ctx0, kv_cache.k,
                                 n_state_head, n_ctx_pad, n_head, n_batch,
                                 ggml_element_size(kv_cache.k)*n_state,
                                 ggml_element_size(kv_cache.k)*n_state_head,
                                 ggml_element_size(kv_cache.k)*n_state*n_ctx_pad,
                                 0);

            struct ggml_tensor * V =
                    ggml_view_4d(ctx0, kv_cache.v,
                                 n_state_head, n_ctx_pad, n_head, n_batch,
                                 ggml_element_size(kv_cache.v)*n_state,
                                 ggml_element_size(kv_cache.v)*n_state_head,
                                 ggml_element_size(kv_cache.v)*n_state*n_ctx_pad,
                                 0);
            ggml_tensor *KQV = ggml_flash_attn_ext(ctx0, Q_h, K, V, nullptr, KQscale, 0.0f, 0.0f);
            cur = ggml_reshape_3d(ctx0, KQV, n_state, n_ctx, n_batch);
        } else{
            // K * Q
            struct ggml_tensor *KQ = ggml_mul_mat(ctx0, K_h, Q_h);

            struct ggml_tensor *KQ_soft_max = ggml_soft_max_ext(ctx0, KQ, nullptr, KQscale, 0.0f);


            ggml_tensor *KQV = ggml_mul_mat(
                    ctx0, ggml_cont(ctx0, ggml_transpose(ctx0, V_h)), KQ_soft_max);
            struct ggml_tensor *KQV_merged = ggml_permute(ctx0, KQV, 0, 2, 1, 3);
            cur = ggml_cpy(ctx0,
                           KQV_merged,
                           ggml_new_tensor_3d(ctx0, GGML_TYPE_F32, n_state, n_ctx, n_batch));
        }

        cur = ggml_add(ctx0, ggml_mul_mat(ctx0, e_attn_ln_out_w, cur),
                       e_attn_ln_out_b);
        ggml_set_name(cur, "attention_out");

        cur = ggml_add(ctx0, cur, fsmn_memory);

        cur = ggml_add(ctx0, cur, fsmn_memory);

        if (e_norm_w1->ne[0] == e_norm_w2->ne[0]) {
            cur = ggml_add(ctx0, cur, residual);
        }
    }

    residual = ggml_cpy(
            ctx0, cur,
            ggml_new_tensor_3d(ctx0, GGML_TYPE_F32, cur->ne[0], cur->ne[1], cur->ne[2]));
    {
        // layer norm after attention
        // cur = ln_0_w*cur + ln_0_b
        cur = ggml_norm(ctx0, cur, params.eps);
        cur = ggml_add(ctx0, ggml_mul(ctx0, cur, e_norm_w2), e_norm_b2);
    }

    {
        // position-wise feed forward layer
        cur = ggml_add(ctx0, ggml_mul_mat(ctx0, e_mlp_w1, cur),
                       e_mlp_b1);
        cur = ggml_relu(ctx0, cur);
        cur = ggml_add(ctx0, ggml_mul_mat(ctx0, e_mlp_w2, cur),
                       e_mlp_b2);
    }
    // residual after position wise feed forward
    cur = ggml_add(ctx0, cur, residual);
    return cur;
}

void sensevoice_encoder::OnLoad(const gguf_loader &loader, const std::string &prefix)
{
    int i = 0;
    LOAD_METADATA(output_size);
    LOAD_METADATA(linear_units);
    LOAD_METADATA(attention_heads);
    LOAD_METADATA(num_blocks);
    LOAD_METADATA(tp_blocks);

    // encoders0
    std::string name = "encoders0." + std::to_string(i);
    LOAD_SUBMODULE_EX(name.c_str(), encoder0);

    //encoders
    encoders_layer.resize(num_blocks);
    for (i = 0; i < num_blocks - 1; i++) 
    {    
        auto& layer = encoders_layer[i];
        name = "encoders." + std::to_string(i);
        LOAD_SUBMODULE_EX(name.c_str(), layer);
    }   

    //tp_encoders
	tp_encoders_layer.resize(tp_blocks);
    for (i = 0; i < tp_blocks; i++) 
    {    
        auto& layer = tp_encoders_layer[i];
        name = "tp_encoders." + std::to_string(i);
        LOAD_SUBMODULE_EX(name.c_str(), layer);
    }   

    LOAD_TENSOR_EX("after_norm.weight",  e_after_norm_w);
    LOAD_TENSOR_EX("after_norm.bias",  e_after_norm_b);

    LOAD_TENSOR_EX("tp_norm.weight",  e_tp_norm_w);
    LOAD_TENSOR_EX("tp_norm.bias",  e_tp_norm_b);
}

void ctc_linear::OnLoad(const gguf_loader &loader, const std::string &prefix)
{
    LOAD_TENSOR_EX("ctc.ctc_lo.weight",  weight);
    LOAD_TENSOR_EX("ctc.ctc_lo.bias",  bias);
}


ggml_tensor *sensevoice_encoder::build_cgraph(ggml_context* ctx, ggml_cgraph* gf, ggml_tensor* x, sensevoice_context_params_t &params, sensevoice_kv_cache &kv_cache, bool use_flash_attn) const 
{
    
    // encoders0 forward
    x = encoder0.build_cgraph(ctx, gf, x, params, kv_cache, use_flash_attn);

    // encoders forward
    for (int i = 0; i < params.n_encoder_layers - 1; i++)
    {
        //LOG_DEBUG("encoder i %d", i);
        auto& layer = encoders_layer[i];
        x = layer.build_cgraph(ctx, gf, x, params, kv_cache, use_flash_attn);
    }

    {
        //after encoder norm
        x = ggml_norm(ctx, x, params.eps);
        x = ggml_add(ctx, ggml_mul(ctx, x, e_after_norm_w), e_after_norm_b);
    }

    // tp encoders forward
    for (int i=0; i < params.n_tp_encoder_layers; i++)
    {
        //LOG_DEBUG("tp encoder i %d", i);
        x = tp_encoders_layer[i].build_cgraph(ctx, gf, x, params, kv_cache, use_flash_attn);
    }

    {
        //after encoder norm
        x = ggml_norm(ctx, x, params.eps);
        x = ggml_add(ctx, ggml_mul(ctx, x, e_after_norm_w), e_after_norm_b);
    }
    return x;
}


int sensevoice_model::init_sensevoice_kv_cache()
{
    int64_t n_text_state = params.n_encoder_hidden_state;
    int64_t n_text_layer = 1;
    int n_ctx = GGML_PAD(params.n_audio_ctx, 256);  //to do n_audio_ctx
    const int64_t n_mem = n_text_layer * n_ctx;
    const int64_t n_elements = n_text_state * n_mem;

    struct ggml_init_params ggml_params = {
        .mem_size = 2 * ggml_tensor_overhead(),
        .mem_buffer = nullptr,
        .no_alloc = true,
    };

    kv_cache.head = 0;
    kv_cache.size = n_ctx;

    kv_cache.cells.clear();
    kv_cache.cells.resize(n_ctx);

    kv_cache.ctx = ggml_init(ggml_params);

    if(!kv_cache.ctx)
    {
        LOG_ERROR("failed to allocate memory for the kv cache context");
        return ERROR;
    }

    kv_cache.k = ggml_new_tensor_1d(kv_cache.ctx, params.wtype, n_elements);
    kv_cache.v = ggml_new_tensor_1d(kv_cache.ctx, params.wtype, n_elements);

    kv_cache.buffer = ggml_backend_alloc_ctx_tensors(kv_cache.ctx, backend.get());
    if(!kv_cache.buffer)
    {
        LOG_ERROR("failed to allocate memory for the kv cache context");
        return ERROR;
    }

    ggml_backend_buffer_clear(kv_cache.buffer, 0);
    return SUCCESS;
}


void sensevoice_model::load(gguf_loader &loader)
{

    init_sensevoice_kv_cache();

    embedding.OnLoad(loader, {});
    encoder.OnLoad(loader, "encoder");

    output.OnLoad(loader, {});

    params.n_encoder_hidden_state = encoder.output_size;
    params.n_encoder_attention_heads = encoder.attention_heads;

    auto tensors = embedding.get_all_tensors();

    for(auto &kv : encoder.get_all_tensors())
        tensors.insert(std::move(kv));

    for(auto &kv : output.get_all_tensors())
        tensors.insert(std::move(kv));

    ggml_init_params params = 
    {
        .mem_size = (tensors.size() + 8) * ggml_tensor_overhead(),
        .mem_buffer = nullptr,
        .no_alloc = true
    };

    auto buft = ggml_backend_get_default_buffer_type(backend.get());
    auto alignment = ggml_backend_buft_get_alignment(buft);
    size_t mem_size = 0;

    for(const auto &[name, tensor] : tensors)
    {   
        mem_size += get_aligned_size(ggml_backend_buft_get_alloc_size(buft, *tensor), alignment);
    }   

    LOG_DEBUG("mem_size %lld ", mem_size);

    buffer.reset(ggml_backend_buft_alloc_buffer(buft, mem_size));
    auto buffer_base = reinterpret_cast<char*>(ggml_backend_buffer_get_base(buffer.get()));

    ctx.reset(ggml_init(params)); 

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

}

void sensevoice_model::build_cgraph()
{
    ggml_tensor *cur = nullptr;
    //auto ctx0 = ctx.get();

    int chunk_size = 20;
    int feature_dim = 560;

    meta.resize(ggml_tensor_overhead() * 8192 + ggml_graph_overhead());

    ggml_init_params gf_params = 
    {
        .mem_size = meta.size(),
        .mem_buffer = meta.data(),
        .no_alloc = true
    };

    ggml_context *ctx0 = ggml_init(gf_params);

    //ggml_reset(ctx0);

    ggml_backend_sched_reset(sched.get());

    //gf = ggml_new_graph(ctx0);
    gf = ggml_new_graph_custom(ctx0, 8192, false);

    struct ggml_tensor *feature = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, feature_dim, chunk_size);
    ggml_set_name(feature, "feats");
    ggml_set_input(feature);
    inputs[0] = feature;

    struct ggml_tensor *embedding = ggml_new_tensor_2d(ctx0, GGML_TYPE_I32, 4, 1); 
    ggml_set_name(embedding, "embedding");
    ggml_set_input(embedding);
    inputs[1] = embedding;

    embedding = ggml_get_rows(ctx0, this->embedding.weight, embedding);
    embedding = ggml_repeat(ctx0, embedding, ggml_new_tensor_3d(ctx0, GGML_TYPE_I32, embedding->ne[0], embedding->ne[1], feature->ne[2]));

    cur = ggml_concat(ctx0, embedding, feature, 1); 

    cur = ggml_scale(ctx0, cur, sqrtf(params.n_encoder_hidden_state));

    // implement encoder small forward graph
    //ref: https://github.com/modelscope/FunASR/blob/b7b4a83c18277a7022124cad790c08ae703b7a2d/funasr/models/sense_voice/model.py#L558-L583
    // [x] 1. sinusoidal position
    // [x] 2. encoders0
    // [x] 3. encoders
    // [x] 4. tp_encoders
    // [x] 5. tp_norm

    ggml_tensor *position = ggml_new_tensor_3d(ctx0, cur->type, cur->ne[0], cur->ne[1], cur->ne[2]);
    ggml_set_name(position, "position");
    ggml_set_input(position);

    cur = ggml_add(ctx0, position, cur);

    cur = encoder.build_cgraph(ctx0, nullptr, cur, params, kv_cache, params.flash_attn);

    LOG_DEBUG("build_exppand");
    ggml_build_forward_expand(gf, cur);
    LOG_DEBUG("build_exppand");
    set_graph_backend(gf, sched.get(), backend.get(), nullptr);

    ggml_backend_sched_synchronize(sched.get());
    ggml_backend_sched_alloc_graph(sched.get(), gf);
    LOG_DEBUG("ggml_graph_n_nodes %d ", ggml_graph_n_nodes(gf));

    ggml_free(ctx0);
}

int sensevoice_model::inference()
{
    struct ggml_tensor *position = ggml_graph_get_tensor(gf, "position");
    struct ggml_tensor *embedding = ggml_graph_get_tensor(gf, "embedding");

    auto n_len = position->ne[1];
    auto dim = position->ne[0];
    auto n_batch = position->ne[2];
    std::vector<float> _position;
     _position.resize(n_len * dim * n_batch);

    // construct position embedding
    // sinusoidal position embedding
    // reference:
    // https://github.com/modelscope/FunASR/blob/45d7aa9004763684fb748ee17942ecba81042201/funasr/models/transformer/embedding.py#L392-L405
    // P_{k,i} = sin(k/10000^(2i/d))  0 < i < d/2
    // p_{k,j} = cos(k/10000^(2j/d))  d/2 < j < d

    for (int b = 0; b < n_batch; b++)
    for (int k = 1; k <= n_len; k++) {
        for (int i = 0; i < dim / 2; i++) { 
        _position[b * n_len * dim + (k - 1) * dim + i] = sinf(k * pow(10000, -2.0 * i / dim));
        _position[b * n_len * dim + (k - 1) * dim + i + dim / 2] =
                cosf(k * pow(10000, -2.0 * i / dim));
        }
    }
    LOG_DEBUG("position %p embedding %p", position, embedding);

    ggml_backend_tensor_set(
            position, _position.data(), 0,
            ggml_nelements(position) * sizeof(float));

    int _embedding[4] = {params.language_id, 1, 2, params.use_itn ? 14 : 15};
    ggml_backend_tensor_set(inputs[1], _embedding, 0, 4*sizeof(int));
      //ggml_graph_dump_dot(gf, NULL, "sense-voice.dot");
      //ggml_backend_sched_set_eval_callback(sched, ctx.params.cb_eval, ctx.params.cb_eval_user_data);
    //if (!ggml_graph_compute_helper(sched, gf, n_threads)) {
            //return false;
    //}
    ggml_backend_sched_graph_compute(sched.get(), gf);
}


