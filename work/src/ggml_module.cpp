#include "base.h"
#include "span_compat.h"
#include "gguf_loader.h"
#include "ggml_module.h"
#include "ggml_cpu_flag.h"

static void split_tensor(ggml_context *ctx, ggml_tensor *tensor, int dim, ggml_tensor **tensors, uint16_t chunks)
{
    GGML_ASSERT(dim >= 0 && dim < GGML_MAX_DIMS);
    GGML_ASSERT(tensor->ne[dim] % chunks == 0);

    int64_t ne[GGML_MAX_DIMS];
    memcpy(ne, tensor->ne, sizeof(ne));
    ne[dim] /= chunks;
    const size_t offset_per_chunk = tensor->nb[dim] * ne[dim];

    for(uint16_t i = 0; i != chunks; i++)
        tensors[i] = ggml_view_4d(
            ctx,
            tensor,
            ne[0],
            ne[1],
            ne[2],
            ne[3],
            tensor->nb[1],
            tensor->nb[2],
            tensor->nb[3],
            i * offset_per_chunk
        );
}

template<uint16_t chunks>
std::array<ggml_tensor *, chunks> split_tensor(ggml_context *ctx, ggml_tensor *tensor, int dim = 0)
{
    std::array<ggml_tensor *, chunks> result;
    split_tensor(ctx, tensor, dim, result.data(), chunks);
    return result;
}

static ggml_tensor *view_tensor(ggml_context *ctx, ggml_tensor *tensor, ggml_tensor *src = nullptr)
{
    auto view = ggml_view_tensor(ctx, tensor);
    view->op = GGML_OP_VIEW;
    view->src[0] = src ? src : tensor;
    return view;
}

static ggml_tensor *concat_tensors(ggml_context *ctx, ggml_tensor **tensors, size_t chunks, int dim, ggml_backend_op_capabilities_t capabilities)
{
    GGML_ASSERT(chunks > 1);
    GGML_ASSERT(dim >= 0 && dim < GGML_MAX_DIMS);

    if(chunks == 2)
    {
        if (tensors[0]->type == GGML_TYPE_F32 && tensors[1]->type == GGML_TYPE_F32
            || tensors[0]->type == GGML_TYPE_I32 && tensors[1]->type == GGML_TYPE_I32 && capabilities.concat_i32)

        {
            return ggml_concat(ctx, tensors[0], tensors[1], dim);
        }
        else if(tensors[0]->type == GGML_TYPE_I32 && tensors[1]->type == GGML_TYPE_I32)
        {
            auto src0 = view_tensor(ctx, tensors[0]);
            auto src1 = view_tensor(ctx, tensors[1]);

            src0->type = GGML_TYPE_F32;
            src1->type = GGML_TYPE_F32;

            auto res = ggml_concat(ctx, src0, src1, dim);
            res = view_tensor(ctx, res);
            res->type = GGML_TYPE_I32;
            return res;
        }
    }

    auto type = tensors[0]->type;
    int64_t ne[GGML_MAX_DIMS];
    memcpy(ne, tensors[0]->ne, sizeof(ne));

    for(const auto tensor : simple_span<ggml_tensor *>(tensors + 1, chunks - 1))
    {
        GGML_ASSERT(tensor->type == type);
        for(int d = 0; d != GGML_MAX_DIMS; ++d)
        {
            if(d == dim)
                continue;

            GGML_ASSERT(tensor->ne[d] == ne[d]);
        }
        ne[dim] += tensor->ne[dim];
    }

    auto result = ggml_new_tensor_4d(ctx, type, ne[0], ne[1], ne[2], ne[3]);
    auto prev_part = result;
    size_t offset = 0;

    for(auto tensor : simple_span<ggml_tensor *>(tensors, chunks))
    {
        auto dest_part = ggml_view_4d(ctx, result, 
                tensor->ne[0], tensor->ne[1], tensor->ne[2], tensor->ne[3],
                result->nb[1], result->nb[2], result->nb[3], offset);
        dest_part->src[0] = prev_part;

        /* GGML may produce incorrect results when copying a non-contiguous tensor,
         * so make it contiguous first */

        if(!ggml_is_contiguous(tensor))
            tensor = ggml_cont(ctx, tensor);

        prev_part = ggml_cpy(ctx, tensor, dest_part);

        offset += tensor->ne[dim] * result->nb[dim];
    }
    return view_tensor(ctx, result, prev_part);
}

template<size_t chunks>
static ggml_tensor *concat_tensors(ggml_context *ctx, std::array<ggml_tensor *,chunks> tensors, int dim, ggml_backend_op_capabilities_t capabilities = {})
{
    return concat_tensors(ctx, tensors.data(), chunks, dim, capabilities);
}


static ggml_tensor *unsqueeze(ggml_context *ctx, ggml_tensor *x, int dim)
{
    if(dim == 0)
        return ggml_view_4d(ctx, x, 1, x->ne[0], x->ne[1], x->ne[2], x->nb[0], x->nb[1], x->nb[2], 0);
    else if(dim == 1)
        return ggml_view_4d(ctx, x, x->ne[0], 1, x->ne[1], x->ne[2], x->nb[1], x->nb[1], x->nb[2], 0);
    else if(dim == 2)
        return ggml_view_4d(ctx, x, x->ne[0], x->ne[1], 1, x->ne[2], x->nb[1], x->nb[2], x->nb[2], 0);
    else 
        GGML_ABORT("unsqueeze: invalid dim %d for 3D tensor", dim);
}

static ggml_tensor *mish(ggml_context *ctx, ggml_tensor *x)
{
    auto out = ggml_softplus(ctx, x);
    out = ggml_tanh(ctx, out);
    out = ggml_mul(ctx, x, out);
    return out;
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

    if (x->ne[0] % pad == 0 || x->ne[0] / pad < n_pad_req) 
    {   
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

#if 0
ggml_tensor *ggml_fft(ggml_context *ctx, ggml_tensor *a, fft_context_t *fft_ctx)
{
    GGML_ASSERT(ggml_is_matrix(a));
    GGML_ASSERT(a->ne[0] == fft_ctx->nfft);
    GGML_ASSERT(ggml_is_contiguous(a));

    //return ggml_custom_4d(ctx, a->type, fft_ctx->nfft, a->ne[1], 2, 1, &a, 1, reinterpret_cast<ggml_custom_op_t>(ggml_fft_op)
}
#endif

void Module::onload(const gguf_loader& loader, const std::string& prefix) {}

void BasicModule::onload(const gguf_loader& loader, const std::string& prefix)
{
    LOAD_TENSOR(weight);
    LOAD_OPTIONAL_TENSOR(bias);
}

ggml_tensor *Linear::build_cgraph(ggml_context *ctx, ggml_tensor *x) const
{
    auto out = ggml_mul_mat(ctx, weight, x);
    if(bias)
        out = ggml_add(ctx, out, bias);
    return out;
}

void LayerNorm::onload(const gguf_loader &loader, const std::string &prefix)
{
    LOAD_OPTIONAL_TENSOR(weight);
    LOAD_OPTIONAL_TENSOR(bias);
}

ggml_tensor* LayerNorm::build_cgraph(ggml_context* ctx, ggml_tensor* x) const
{
    x = ggml_norm(ctx, x, eps);
    if (weight)
    {   
        x = ggml_mul(ctx,  x, weight); 
        if (bias)
            x = ggml_add(ctx, x, bias);
    }   
    return x;
}

ggml_tensor* LayerNorm::build_cgraph(ggml_context* ctx, ggml_tensor* x, float _eps) const
{
    x = ggml_norm(ctx, x, _eps);
    if (weight)
    {   
        x = ggml_mul(ctx,  x, weight); 
        if (bias)
            x = ggml_add(ctx, x, bias);
    }   
    return x;
}

ggml_tensor* Conv1d::build_cgraph(ggml_context* ctx, ggml_tensor* x, int s, int p, int d, int g, ggml_backend_op_capabilities_t capabilities) const
{
    GGML_ASSERT(g >= 1);
    GGML_ASSERT(x->ne[3] == 1 && weight->ne[3] == 1);

    ggml_tensor* im2col;
    ggml_tensor* weight = this->weight;
    // Backends without F16 IM2COL support (e.g. Metal) need F32 input + F32/F16
    // output and contiguous src[1]. Fall back only when the probe says so.
    const bool needs_f32_fallback = !capabilities.im2col_f16;
    if (needs_f32_fallback && x->type == GGML_TYPE_F16)
        x = ggml_cast(ctx, x, GGML_TYPE_F32);
    if (g != 1)
    {
        GGML_ASSERT(weight->ne[2] % g == 0 && weight->ne[1] * g == x->ne[1]);

        auto xs = reinterpret_cast<ggml_tensor**>(alloca(sizeof(ggml_tensor*) * g));
        auto ws = reinterpret_cast<ggml_tensor**>(alloca(sizeof(ggml_tensor*) * g));
        if (needs_f32_fallback && weight->type == GGML_TYPE_F16)
            weight = ggml_cast(ctx, weight, GGML_TYPE_F32);
        split_tensor(ctx, x, 1, xs, g);
        split_tensor(ctx, weight, 2, ws, g);
        const auto out_type = needs_f32_fallback ? GGML_TYPE_F32 : weight->type;
        for (int i = 0; i != g; i++)
            xs[i] = unsqueeze(ctx,
                ggml_im2col(ctx,
                    ws[i],
                    ggml_cont(ctx, xs[i]),
                    s, 0, p, 0, d, 0,
                    false, out_type),
                2);
        im2col = concat_tensors(ctx, xs, g, 2, capabilities);
    }
    else
    {
        const auto out_type = needs_f32_fallback
            ? ((weight->type == GGML_TYPE_F16) ? GGML_TYPE_F16 : GGML_TYPE_F32)
            : weight->type;
        im2col = ggml_im2col(ctx, weight, x, s, 0, p, 0, d, 0, false, out_type);
        if (im2col->ne[1] > 0xFFFF)
        {
            // Split the im2col output along the output-width dimension.
            // Keep the op as GGML_OP_IM2COL until graph allocation, then switch to
            // GGML_OP_NONE. This avoids a GGML allocation-size bug that can reserve
            // much more memory than necessary.
            constexpr int chunk_max_size = 65528;
            auto prev_chunk = im2col;
            memset(im2col->op_params, 0, sizeof(im2col->op_params));
            memset(im2col->src, 0, sizeof(im2col->src));

            for (int64_t start = 0; start < im2col->ne[1]; start += chunk_max_size)
            {
                auto end = std::min(im2col->ne[1], start + chunk_max_size);

                auto logical_start = start * s - p;
                auto logical_end = (end - 1) * s - p + d * (weight->ne[0] - 1) + 1;

                int64_t phys_start = std::max<int64_t>(0, logical_start);
                int64_t phys_end = std::min(x->ne[0], logical_end);
                int64_t chunk_iw = phys_end - phys_start;

                int chunk_p0 = std::max(0, -static_cast<int>(logical_start));

                auto x_view = ggml_view_3d(
                    ctx, x,
                    chunk_iw, x->ne[1], x->ne[2],
                    x->nb[1], x->nb[2],
                    phys_start * x->nb[0]
                );

                auto chunk_im2col = ggml_im2col(
                    ctx, weight,
                    needs_f32_fallback ? ggml_cont(ctx, x_view) : x_view,
                    s, 1,
                    chunk_p0, 0,
                    d, 1,
                    false,
                    im2col->type
                );

                chunk_im2col->view_src = im2col;
                chunk_im2col->view_offs = start * im2col->nb[1];
                chunk_im2col->src[2] = prev_chunk;
                prev_chunk = chunk_im2col;
            }

            im2col = view_tensor(ctx, im2col, prev_chunk);
        }
    }

    weight = ggml_reshape_3d(ctx, weight, weight->ne[0] * weight->ne[1], weight->ne[2] / g, g);
    if (x->ne[2] != 1)
        if (weight->type == GGML_TYPE_F16 && !capabilities.repeat_f16)
        {
            weight->ne[3] = x->ne[2];
            weight->nb[3] = 0;
        }
        else
            weight = ggml_repeat_4d(ctx, weight, weight->ne[0], weight->ne[1], g, x->ne[2]);

    ggml_tensor* result = ggml_mul_mat(ctx, im2col, weight);
    result = ggml_reshape_3d(ctx, result, im2col->ne[1], this->weight->ne[2], x->ne[2]);

    if (bias)
        result = ggml_add(ctx, result, unsqueeze(ctx, bias, 0));

    return result;
}
#if 0

int CausalConv1d::causal_padding() const
{
    const auto kernel_size = weight->ne[0];
    const auto causal_padding = static_cast<int>((kernel_size * d - d) / 2 * 2 + (kernel_size + 1) % 2);
    return causal_padding;
}

ggml_tensor *CausalConv1d::build_cgraph(ggml_context *ctx, ggml_tensor *x) const
{
    if(!ggml_is_contiguous(x))
        x = ggml_cont(ctx, x);
    if(left == causal_type)
        x = ggml_pad_ext(ctx, x, causal_padding(), 0, 0, 0, 0, 0, 0, 0);
    else if(right == causal_type)
        x = ggml_pad_ext(ctx, x, 0, causal_padding(), 0, 0, 0, 0, 0, 0);
    else 
        GGML_ABORT("CausalConv1d: invalid causal type %d", static_cast<int>(causal_type));

    x = Conv1d::build_cgraph(ctx, x, 1, 0, d, 1, {});
    return x;
}

CausalConvRNNF0Predictor::CausalConvRNNF0Predictor()
{
    condnet_0.causal_type = CausalConv1d::right;
    condnet_2.causal_type = CausalConv1d::left;
    condnet_4.causal_type = CausalConv1d::left;
    condnet_6.causal_type = CausalConv1d::left;
    condnet_8.causal_type = CausalConv1d::left;
}

void CausalConvRNNF0Predictor::onload(const gguf_loader &loader, const std::string &prefix)
{
    LOAD_SUBMODULE_EX("condnet.0", condnet_0);
    LOAD_SUBMODULE_EX("condnet.2", condnet_0);
    LOAD_SUBMODULE_EX("condnet.4", condnet_0);
    LOAD_SUBMODULE_EX("condnet.6", condnet_0);
    LOAD_SUBMODULE_EX("condnet.8", condnet_0);
    LOAD_SUBMODULE(classifier);
}

ggml_tensor *CausalConvRNNF0Predictor::build_cgraph(ggml_context *ctx, ggml_tensor *x) const
{
    x = condnet_0.build_cgraph(ctx, x);
    x = ggml_elu(ctx, x);
    x = condnet_2.build_cgraph(ctx, x);
    x = ggml_elu(ctx, x);
    x = condnet_4.build_cgraph(ctx, x);
    x = ggml_elu(ctx, x);
    x = condnet_6.build_cgraph(ctx, x);
    x = ggml_elu(ctx, x);
    x = condnet_8.build_cgraph(ctx, x);
    x = ggml_elu(ctx, x);

    x = ggml_permute(ctx, x, 1, 0, 2, 3);
    x = ggml_cont(ctx, x);
    x = classifier.build_cgraph(ctx, x);
    x = ggml_reshape_2d(ctx, x, x->ne[1], x->ne[2]);

    return ggml_abs(ctx, x);
}
#endif

int CausalConv1d::causal_padding() const
{
    const auto kernel_size = weight->ne[0];
    const auto causal_padding = static_cast<int>((kernel_size * d - d) / 2 * 2 + (kernel_size + 1) % 2);
    return causal_padding;
}

ggml_tensor* CausalConv1d::build_cgraph(ggml_context* ctx, ggml_tensor* x) const
{
    if (!ggml_is_contiguous(x))
        x = ggml_cont(ctx, x);
    if (causal_type == left)
        x = ggml_pad_ext(ctx, x, causal_padding(), 0, 0, 0, 0, 0, 0, 0);
    else if (causal_type == right)
        x = ggml_pad_ext(ctx, x, 0, causal_padding(), 0, 0, 0, 0, 0, 0);
    else
        GGML_ABORT("CausalConv1d: invalid causal type %d", static_cast<int>(causal_type));

    x = Conv1d::build_cgraph(ctx, x, 1, 0, d, 1, {});
    return x;
}

void CausalConvRNNF0Predictor::onload(const gguf_loader& loader, const std::string& prefix)
{
    LOAD_SUBMODULE_EX("condnet.0", condnet_0);
    LOAD_SUBMODULE_EX("condnet.2", condnet_2);
    LOAD_SUBMODULE_EX("condnet.4", condnet_4);
    LOAD_SUBMODULE_EX("condnet.6", condnet_6);
    LOAD_SUBMODULE_EX("condnet.8", condnet_8);
    LOAD_SUBMODULE(classifier);
}


CausalConvRNNF0Predictor::CausalConvRNNF0Predictor()
{
    condnet_0.causal_type = CausalConv1d::right;
    condnet_2.causal_type = CausalConv1d::left;
    condnet_4.causal_type = CausalConv1d::left;
    condnet_6.causal_type = CausalConv1d::left;
    condnet_8.causal_type = CausalConv1d::left;
}

ggml_tensor* CausalConvRNNF0Predictor::build_cgraph(ggml_context* ctx, ggml_tensor* x) const
{
    x = condnet_0.build_cgraph(ctx, x);
    x = ggml_elu(ctx, x);
    x = condnet_2.build_cgraph(ctx, x);
    x = ggml_elu(ctx, x);
    x = condnet_4.build_cgraph(ctx, x);
    x = ggml_elu(ctx, x);
    x = condnet_6.build_cgraph(ctx, x);
    x = ggml_elu(ctx, x);
    x = condnet_8.build_cgraph(ctx, x);
    x = ggml_elu(ctx, x);

    x = ggml_permute(ctx, x, 1, 0, 2, 3);
    x = ggml_cont(ctx, x);
    x = classifier.build_cgraph(ctx, x);
    x = ggml_reshape_2d(ctx, x, x->ne[1], x->ne[2]);
    return ggml_abs(ctx, x);
}

ggml_tensor *CausalConv1dDownSample::build_cgraph(ggml_context *ctx, ggml_tensor *x) const 
{
    x = ggml_pad_ext(ctx, x, s - 1, 0, 0, 0, 0, 0, 0, 0);
    return Conv1d::build_cgraph(ctx, x, s, 0, 1, 1, {});
}

ggml_tensor *CausalConv1dUpSample::build_cgraph(ggml_context *ctx, ggml_tensor *x) const 
{
    x = ggml_interpolate(ctx, x, x->ne[0] * s, x->ne[1], x->ne[2], 1, GGML_SCALE_MODE_NEAREST);
    x = ggml_pad_ext(ctx, x, static_cast<int>(weight->ne[0] - 1), 0, 0, 0, 0, 0, 0, 0);
    return Conv1d::build_cgraph(ctx, x, 1, 0, 1, 1, {});
}

void CausalConvPositionEmbedding::onload(const gguf_loader &loader, const std::string &prefix)
{
    LOAD_SUBMODULE_EX("conv1.0", conv1);
    LOAD_SUBMODULE_EX("conv2.0", conv2);
}

ggml_tensor *CausalConvPositionEmbedding::build_cgraph(ggml_context *ctx, ggml_tensor *x, ggml_backend_op_capabilities_t capabilities) const
{
    x = ggml_permute(ctx, x, 1, 0, 2, 3);
    x = ggml_cont(ctx, x);
    x = ggml_pad_ext(ctx, x, static_cast<int>(conv1.weight->ne[0] - 1), 0, 0, 0, 0, 0, 0, 0);
    x = conv1.build_cgraph(ctx, x, 1, 0, 1, 16, capabilities);
    x = mish(ctx, x);
    x = ggml_pad_ext(ctx, x, static_cast<int>(conv2.weight->ne[0] - 1), 0, 0, 0, 0, 0, 0, 0);
    x = conv2.build_cgraph(ctx, x, 1, 0, 1, 16, capabilities);
    x = mish(ctx, x);

    return ggml_permute(ctx, x, 1, 0, 2, 3);
}

void Snake::onload(const gguf_loader &loader, const std::string &prefix)
{
    LOAD_TENSOR(alpha);

    constexpr float epsilon = 0.000000001f;
    for(auto &i : simple_span<float>(reinterpret_cast<float *>(ggml_get_data(alpha)), ggml_nelements(alpha)))
    {
        if(std::abs(i) < epsilon)
            i = epsilon;
    }
}

ggml_tensor *Snake::build_cgraph(ggml_context *ctx, ggml_tensor *x) const
{
    auto alpha = unsqueeze(ctx, this->alpha, 0);
    auto x_sin = ggml_sin(ctx, ggml_mul(ctx, x, alpha));
    x_sin = ggml_mul(ctx, x_sin, x_sin);
    return ggml_add(ctx, x, ggml_div(ctx, x_sin, alpha));
}

void ResBlock::onload(const gguf_loader &loader, const std::string &prefix)
{
    int64_t id;
    GGML_ASSERT(loader.find_metadata_key(combine_prefix(prefix, "dilations").c_str(), id));
    GGML_ASSERT(gguf_get_arr_type(loader.gguf_ctx, id) == GGUF_TYPE_INT32);

    const auto n_dilations = gguf_get_arr_n(loader.gguf_ctx, id);
    const int *dilations = reinterpret_cast<const int *>(gguf_get_arr_data(loader.gguf_ctx, id));
    convs.resize(n_dilations);
    for(size_t i = 0; i != n_dilations; ++i)
    {
        auto &[snk1, conv1, snk2, conv2] = convs[i];
        conv1.causal_type = CausalConv1d::causal_type_t::left;
        conv1.d = dilations[i];
        conv2.causal_type = CausalConv1d::causal_type_t::left;
        conv2.d = 1;

        LOAD_SUBMODULE_EX(("convs1." + std::to_string(i)).c_str(), conv1);
        LOAD_SUBMODULE_EX(("convs2." + std::to_string(i)).c_str(), conv2);
        LOAD_SUBMODULE_EX(("activations1." + std::to_string(i)).c_str(), snk1);
        LOAD_SUBMODULE_EX(("activations2." + std::to_string(i)).c_str(), snk2);
    }
}

ggml_tensor *ResBlock::build_cgraph(ggml_context *ctx, ggml_tensor *x) const
{
    for(const auto &[act1, conv1, act2, conv2] : convs)
    {
        auto xt = act1.build_cgraph(ctx, x);
        xt = conv1.build_cgraph(ctx, xt);
        xt = act2.build_cgraph(ctx, xt);
        xt = conv2.build_cgraph(ctx, xt);
        x = ggml_add(ctx, x, xt);
    }
    return x;
}

void STFT::onload(const gguf_loader& loader, const std::string& prefix)
{
    LOAD_TENSOR_EX("stft.forward_basis_buffer.weight", forward_basis_buffer);
}

ggml_tensor *STFT::build_cgraph(ggml_context *ctx, ggml_tensor *x) const
{
    x = ggml_conv_1d(ctx, forward_basis_buffer, x, 128, 0, 1);

    ggml_tensor *real_part = ggml_view_2d(ctx, x, x->ne[0], x->ne[1] / 2, x->nb[1], 0);
    ggml_set_name(real_part, "real_part");

    ggml_tensor *image_part = ggml_view_2d(ctx, x, x->ne[0], x->ne[1] / 2, x->nb[1], x->nb[0] * x->ne[0] * x->ne[1] / 2);
    ggml_set_name(image_part, "image_part");

    x = ggml_sqrt(ctx,
                    ggml_add(ctx,
                            ggml_mul(ctx, real_part, real_part),
                            ggml_mul(ctx, image_part, image_part)
                        )
                );
    ggml_set_name(x, "magnitude");
    return x;
}

void LSTM::onload(const gguf_loader &loader, const std::string &prefix)
{
    //to do 内存释放问题
    LOAD_TENSOR_EX("rnn.weight_ih", lstm_weight_ih);
    LOAD_TENSOR_EX("rnn.bias_ih", lstm_bias_ih);

    LOAD_TENSOR_EX("rnn.weight_hh", lstm_weight_hh);
    LOAD_TENSOR_EX("rnn.bias_hh", lstm_bias_hh);
}

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
ggml_tensor *LSTM::build_cgraph(ggml_context *ctx, ggml_tensor *x) const
{
    struct ggml_tensor* in_lstm_hidden_state = ggml_new_tensor_1d(ctx, x->type, x->ne[1]);
    struct ggml_tensor*  in_lstm_context = ggml_new_tensor_1d(ctx, x->type, x->ne[1]);

    struct ggml_tensor* out_lstm_hidden_state;
    struct ggml_tensor*  out_lstm_context;

    ggml_set_name(in_lstm_context, "in_lstm_context");
    ggml_set_name(in_lstm_hidden_state, "in_lstm_hidden_state");

    x = ggml_cont(ctx, ggml_transpose(ctx, x));

    struct ggml_tensor *gates = ggml_add(ctx,
                            ggml_add(ctx, ggml_mul_mat(ctx,lstm_weight_ih,x),lstm_bias_ih),
                            ggml_add(ctx, ggml_mul_mat(ctx,lstm_weight_hh,in_lstm_hidden_state),
                                lstm_bias_hh));
    ggml_set_name(gates, "gates");

    struct ggml_tensor * input_gates = ggml_sigmoid(ctx, ggml_view_2d(ctx, gates, gates->ne[0] / 4, gates->ne[1] , gates->nb[1], 0));
    struct ggml_tensor * forget_gates = ggml_sigmoid(ctx, ggml_view_2d(ctx, gates, gates->ne[0] / 4, gates->ne[1], gates->nb[1], gates->nb[0] / 4 * gates->ne[0]));
    struct ggml_tensor * cell_gate = ggml_tanh(ctx, ggml_view_2d(ctx, gates, gates->ne[0] / 4, gates->ne[1], gates->nb[1], 2 * gates->nb[0] / 4 * gates->ne[0]));
    struct ggml_tensor * out_gates = ggml_sigmoid(ctx, ggml_view_2d(ctx, gates, gates->ne[0] / 4, gates->ne[1], gates->nb[1], 3 * gates->nb[0] / 4 * gates->ne[0]));

    ggml_set_name(input_gates, "input_gates");
    ggml_set_name(forget_gates, "forget_gates");
    ggml_set_name(cell_gate, "cell_gates");
    ggml_set_name(out_gates, "out_gates");

    out_lstm_context = ggml_add(ctx,
                                ggml_mul(ctx, forget_gates, in_lstm_context),
                                ggml_mul(ctx, input_gates, cell_gate));

    ggml_set_name(out_lstm_context, "out_lstm_context");
    ggml_set_output(out_lstm_context);
    out_lstm_hidden_state = ggml_mul(ctx, out_gates, ggml_tanh(ctx, out_lstm_context));
	ggml_set_name(out_lstm_hidden_state, "out_lstm_hidden_state");
	ggml_set_output(out_lstm_hidden_state);

	return out_lstm_hidden_state;
}

void SinusoidalPositionEncoder::onload(const gguf_loader& loader, const std::string& prefix)
{
    LOAD_TENSOR(weight);    
}

// construct position embedding
// sinusoidal position embedding
// reference:
//https://github.com/modelscope/FunASR/blob/45d7aa9004763684fb748ee17942ecba81042201/funasr/models/transformer/embedding.py#L392-L405
// P_{k,i} = sin(k/10000^(2i/d))  0 < i < d/2
// p_{k,j} = cos(k/10000^(2j/d))  d/2 < j < d
void SinusoidalPositionEncoder::compute(int language_id, int use_itn, ggml_tensor *position, ggml_tensor *embedding)
{
    //todo
    auto n_len = position->ne[1];
    auto dim = position->ne[0];
    auto n_batch = position->ne[2];
    std::vector<float> _position;
    _position.resize(n_len * dim * n_batch);

    //SIMD to do 
    for(int b = 0; b < n_batch; b++)
    {
        for(int k = 1; k <= n_len; k++)
        {
            for(int i = 0; i < dim / 2; i++)
            {
                _position[b * n_len * dim + (k - 1) * dim + i ] = 
                            sinf(k * pow(10000, -2.0 * i / dim));

                _position[b * n_len * dim + (k - 1) * dim + i  + dim / 2] = 
                            cosf(k * pow(10000, -2.0 * i / dim));
            }
        }
    }
    ggml_backend_tensor_set(
            position, _position.data(), 0,
            ggml_nelements(position) * sizeof(float));

    int _embedding[4] = {language_id, 1, 2, use_itn ? 14 : 15}; 
    ggml_backend_tensor_set(embedding, _embedding, 0, 4 * sizeof(int));
}

ggml_tensor *SinusoidalPositionEncoder::build_cgraph(ggml_context *ctx, ggml_tensor *feature, int n_hidden_state) const
{
    ggml_tensor *embedding = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, 4, 1);
    ggml_set_name(embedding, "embedding");
    ggml_set_input(embedding);

    embedding = ggml_get_rows(ctx, weight, embedding);
    embedding = ggml_repeat(ctx, embedding, ggml_new_tensor_3d(ctx, GGML_TYPE_I32, embedding->ne[0], embedding->ne[1], feature->ne[2]));

    ggml_tensor *x = ggml_concat(ctx, embedding, feature, 1);

    x = ggml_scale(ctx, x, sqrt(n_hidden_state));

    ggml_tensor *position = ggml_new_tensor_3d(ctx, x->type, x->ne[0], x->ne[1], x->ne[2]);
    ggml_set_name(position, "position");
    ggml_set_input(position);

    x = ggml_add(ctx, position, x);

    return x;
}

void PositionwiseFeedForward::onload(const gguf_loader& loader, const std::string& prefix)
{
    LOAD_SUBMODULE(w_1);
    LOAD_SUBMODULE(w_2);
}

ggml_tensor *PositionwiseFeedForward::build_cgraph(ggml_context *ctx, ggml_tensor *x) const
{
    x = ggml_add(ctx, ggml_mul_mat(ctx, w_1.weight, x), w_1.bias);
    x = ggml_relu(ctx, x);
    x = ggml_add(ctx, ggml_mul_mat(ctx, w_2.weight, x), w_2.bias);
    return x;
}

void InputEmbedding::onload(const gguf_loader &loader, const std::string &prefix)
{
    LOAD_SUBMODULE(proj);
    LOAD_SUBMODULE(conv_pos_embed);
}

ggml_tensor *InputEmbedding::build_cgraph(ggml_context *ctx, ggml_tensor *x, ggml_tensor *cond, ggml_tensor *text_embed, ggml_tensor *spks, ggml_backend_op_capabilities_t capabilities) const 
{
    spks = ggml_repeat(ctx, spks, x);
    x = concat_tensors(ctx, std::array{x, cond, text_embed, spks}, 0);
    x = proj.build_cgraph(ctx, x);
    x = ggml_add(ctx,
            ggml_cont(ctx, conv_pos_embed.build_cgraph(ctx, x, capabilities)),
            x);
    return x;
}

void TimestepEmbedding::onload(const gguf_loader &loader, const std::string &prefix)
{
    LOAD_SUBMODULE_EX("time_mlp.0", time_mlp_0);
    LOAD_SUBMODULE_EX("time_mlp.2", time_mlp_2);
}

ggml_tensor *TimestepEmbedding::build_cgraph(ggml_context *ctx, ggml_tensor *t) const
{
    auto time_hidden = time_embed.build_cgraph(ctx, t);
    auto time = time_mlp_0.build_cgraph(ctx, time_hidden);
    time = ggml_silu(ctx, time);
    time = time_mlp_2.build_cgraph(ctx, time);
    return time;
}

void CTC::onload(const gguf_loader &loader, const std::string &prefix)
{
    LOAD_OPTIONAL_TENSOR(weight);
    LOAD_OPTIONAL_TENSOR(bias);
}

std::array<ggml_tensor *, 2> CTC::build_cgraph(ggml_context *ctx, ggml_tensor *encoder_out) const
{
    // Reshape encoder_out to merge batch and time dimensions
    ggml_tensor *x;
    {
        x = ggml_reshape_2d(ctx, encoder_out, encoder_out->ne[0], encoder_out->ne[1] * encoder_out->ne[2]);
        x = ggml_mul_mat(ctx, weight, x);
        x = ggml_add(ctx, x, bias);
        // Reshape back to 3D
        x = ggml_reshape_3d(ctx, x, x->ne[0], encoder_out->ne[1], encoder_out->ne[2]);
    }

    ggml_tensor * probs = ggml_soft_max(ctx, x);
    probs = ggml_reshape_2d(ctx, probs, probs->ne[0], probs->ne[1] * probs->ne[2] * probs->ne[3]);
    ggml_tensor * argmax_logit = ggml_argmax(ctx, probs);
    argmax_logit = ggml_reshape_3d(ctx, argmax_logit, x->ne[1], x->ne[2], x->ne[3]);
    return {probs, argmax_logit};
}

void AdaLayerNormFinal::onload(const gguf_loader &loader, const std::string &prefix)
{
    LOAD_SUBMODULE(linear);
    LOAD_SUBMODULE(norm);
}

ggml_tensor *AdaLayerNormFinal::build_cgraph(ggml_context *ctx, ggml_tensor *x, ggml_tensor *emb) const
{
    emb = ggml_silu(ctx, emb);
    emb = linear.build_cgraph(ctx, emb);
    auto [scale, shift] = split_tensor<2>(ctx, emb, 0);

    scale = ggml_scale_bias(ctx, ggml_cont(ctx, scale), 1.f, 1.f);
    x = norm.build_cgraph(ctx, x);
    x = ggml_mul(ctx, x, unsqueeze(ctx, scale, 1));
    x = ggml_add(ctx, x, unsqueeze(ctx, shift, 1));
    return x;
}

void AdaLayerNormZero::onload(const gguf_loader &loader, const std::string &prefix)
{
    LOAD_SUBMODULE(linear);
    LOAD_SUBMODULE(norm);
}

std::array<ggml_tensor *, 5>AdaLayerNormZero::build_cgraph(ggml_context *ctx, ggml_tensor *x, ggml_tensor *emb) const
{
    emb = ggml_silu(ctx, emb);
    emb = linear.build_cgraph(ctx, emb);
    auto [shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp] = split_tensor<6>(ctx, emb, 0);

    scale_msa = ggml_scale_bias(ctx, ggml_cont(ctx, scale_msa), 1.f, 1.f);

    x = norm.build_cgraph(ctx, x);
    x = ggml_mul(ctx, x, unsqueeze(ctx, scale_msa, 1));
    x = ggml_add(ctx, x, unsqueeze(ctx, shift_msa, 1));
    return {x, gate_msa, shift_mlp, scale_mlp, gate_mlp };
}

void Attention::onload(const gguf_loader &loader, const std::string &prefix)
{
    LOAD_SUBMODULE(to_q);
    LOAD_SUBMODULE(to_k);
    LOAD_SUBMODULE(to_v);
    LOAD_SUBMODULE_EX("to_out.0", to_out);
}

ggml_tensor *Attention::build_cgraph(ggml_context *ctx, ggml_tensor *x, ggml_tensor *position_ids, int64_t cut_len) const
{
    const auto full_seq_len = position_ids->ne[0];
    const auto seq_len = full_seq_len - (cut_len > 0 ? cut_len : 0);
    const auto batch_size = x->ne[2];

    auto key = to_k.build_cgraph(ctx, x);
    auto value = to_v.build_cgraph(ctx, x);

    auto full_position_ids = position_ids;
    if(cut_len > 0)
    {
        x = ggml_view_3d(ctx, x, x->ne[0], x->ne[1] - cut_len, x->ne[2], x->nb[1], x->nb[2], x->nb[1] * cut_len);
        position_ids = ggml_view_2d(ctx, position_ids, position_ids->ne[0] - cut_len, position_ids->ne[1], position_ids->nb[1], position_ids->nb[0] * cut_len);
        position_ids = ggml_cont(ctx, position_ids);
    }
    auto query = to_q.build_cgraph(ctx, x);

    /* Follow the original DiT implementation and apply RoPE before reshaping and 
     * permuting. The ggml RoPE op does not support 4D tensors yet, so reshape to
     * 3D with an explicit single-head dimension first.
     */
    query = ggml_reshape_3d(ctx, query, query->ne[0], 1, seq_len * batch_size);
    key = ggml_reshape_3d(ctx, key, key->ne[0], 1, full_seq_len * batch_size);
    position_ids = ggml_reshape_1d(ctx, position_ids, seq_len * batch_size);
    full_position_ids = ggml_reshape_1d(ctx, full_position_ids, full_seq_len * batch_size);

    const auto head_dim = static_cast<int>(key->ne[0] / heads);
    query = ggml_rope(ctx, query, position_ids, head_dim, GGML_ROPE_TYPE_NORMAL);
    key = ggml_rope(ctx, key, full_position_ids, head_dim, GGML_ROPE_TYPE_NORMAL);

    query = ggml_reshape_4d(ctx, query, head_dim, heads, seq_len, batch_size);
    key = ggml_reshape_4d(ctx, key, head_dim, heads, full_seq_len, batch_size);
    value = ggml_reshape_4d(ctx, value, value->ne[0] / heads, heads, full_seq_len, batch_size);

    query = ggml_permute(ctx, query, 0, 2, 1, 3);
    key = ggml_permute(ctx, key, 0, 2, 1, 3);
    value = ggml_permute(ctx, value, 0, 2, 1, 3);

    query = ggml_cont(ctx, query);
    key = ggml_cont(ctx, key);

    ggml_tensor *attn_output;

    if(fattn)
        attn_output = ggml_flash_attn_ext(ctx, query, key, value, nullptr, 1.f / sqrtf(static_cast<float>(head_dim)), 0.f, 0.f);
    else
    {
        auto attn_scores = ggml_mul_mat(ctx, key, query);
        attn_scores = ggml_scale(ctx, attn_scores, 1.f / sqrtf(static_cast<float>(head_dim)));
        auto attn_weights = ggml_soft_max(ctx, attn_scores);
        value = ggml_permute(ctx, value, 1, 0, 2, 3);
        value = ggml_cont(ctx, value);
        attn_output = ggml_mul_mat(ctx, value, attn_weights);
        attn_output = ggml_permute(ctx, attn_output, 0, 2, 1, 3);
        attn_output = ggml_cont(ctx, attn_output);
    }
    x = ggml_reshape_3d(ctx,
            attn_output,
            attn_output->ne[0] * attn_output->ne[1],
            seq_len,
            batch_size);

    x = to_out.build_cgraph(ctx, x);
    return x;
}

void MultiHeadedAttentionSANM::onload(const gguf_loader& loader, const std::string& prefix)
{
    LOAD_SUBMODULE(linear_out);

    LOAD_SUBMODULE(linear_q);
    LOAD_SUBMODULE(linear_k);
    LOAD_SUBMODULE(linear_v);

    LOAD_SUBMODULE(fsmn_block);
}

// self attention linear qkv
//      cur = ggml_transpose(ctx0, cur);
// split qkv into separate tensors
// q, k, v = torch.split(q_k_v, int(self.h * self.d_k), dim=-1)
//  ref:
//  https://github.com/alibaba-damo-academy/FunASR/blob/main/funasr/modules/attention.py#L391-L396
ggml_tensor *MultiHeadedAttentionSANM::build_cgraph(ggml_context *ctx, ggml_tensor *x, int n_hidden_state, int n_head, int fsmn_kerenl_size, int flash_attn) const
{
    ggml_tensor *Q, *Q_h;
    ggml_tensor *K, *K_h;
    ggml_tensor *V, *V_h;

    int n_ctx = x->ne[1];
    int n_batch = x->ne[2];

    Q = ggml_add(ctx, ggml_mul_mat_pad(ctx, linear_q.weight, x), linear_q.bias);
    Q_h = ggml_reshape_4d(ctx, Q, n_hidden_state / n_head, n_head, n_ctx, n_batch);
    Q_h = ggml_permute(ctx, Q_h, 0, 2, 1, 3);
    Q_h = ggml_cont(ctx, Q_h);
    ggml_set_name(Q_h, "attention_Q");

    K = ggml_add(ctx, ggml_mul_mat(ctx, linear_k.weight, x), linear_k.bias);
    K_h = ggml_reshape_4d(ctx, K, n_hidden_state / n_head, n_head, n_ctx, n_batch);
    K_h = ggml_permute(ctx, K_h, 0, 2, 1, 3);
    K_h = ggml_cont(ctx, K_h);
    ggml_set_name(K_h, "attention_K");

    V = ggml_add(ctx, ggml_mul_mat(ctx, linear_v.weight, x), linear_v.bias);
    V_h = ggml_reshape_4d(ctx, V, n_hidden_state / n_head, n_head, n_ctx, n_batch);
    V_h = ggml_permute(ctx, V_h, 0, 2, 1, 3);
    V_h = ggml_cont(ctx, V_h);
    ggml_set_name(Q_h, "attention_V");

    /* fsmn forward with V */
    int padding = (fsmn_kerenl_size - 1) / 2;
    ggml_tensor *fsmn_memory = nullptr;
    // conv depth wise to do 
    {
        // implement conv depth wise with groups=input_channel implement
        // same in pytorch : F.conv1d(input, weight, bias=None, stride=1, padding=1, dilation=1, grous=n_stae)
        {
            ggml_tensor *a = fsmn_block.weight;
            ggml_tensor *b = ggml_cont(ctx, ggml_transpose(ctx, V));

            ggml_tensor *im2col = ggml_im2col(ctx, a, ggml_reshape_4d(ctx, b, b->ne[0], 1, b->ne[1] * b->ne[2], b->ne[3]), 1, 0, padding, 0, 1, 0, false, GGML_TYPE_F32);
            im2col = ggml_reshape_4d(ctx, im2col, im2col->ne[0], im2col->ne[1], im2col->ne[2] / n_batch, n_batch);
            a = ggml_repeat(ctx, ggml_cast(ctx, a, GGML_TYPE_F32), ggml_new_tensor_4d(ctx, GGML_TYPE_F16, a->ne[0], a->ne[1], a->ne[2], n_batch));
            ggml_tensor *result = ggml_mul_mat(ctx, a, im2col);
            fsmn_memory = ggml_reshape_3d(ctx, result, im2col->ne[1], im2col->ne[2], im2col->ne[3]);
        }
        fsmn_memory = ggml_cont(ctx, ggml_transpose(ctx, fsmn_memory));
        fsmn_memory = ggml_add(ctx, fsmn_memory, V);
        ggml_set_name(fsmn_memory, "fsmn_memory");
    }
    float KQscale = 1.0f / sqrt(float(n_hidden_state) / n_head);

    if(flash_attn)
    {
        //kv_cache .... to do 
    }
    else
    {
        // K * Q
        ggml_tensor *KQ = ggml_mul_mat(ctx, K_h, Q_h);
        ggml_tensor *KQ_soft_max = ggml_soft_max_ext(ctx, KQ, nullptr, KQscale, 0.0f);

        ggml_tensor *KQV = ggml_mul_mat(ctx, ggml_cont(ctx, ggml_transpose(ctx, V_h)), KQ_soft_max);

        ggml_tensor *KQV_merged = ggml_permute(ctx, KQV, 0, 2, 1, 3);

        x = ggml_cpy(ctx, KQV_merged, ggml_new_tensor_3d(ctx, GGML_TYPE_F32, n_hidden_state, n_ctx, n_batch));
    }
    x = ggml_add(ctx, ggml_mul_mat(ctx, linear_out.weight, x), linear_out.bias);

    ggml_set_name(x, "attention_out");

    x = ggml_add(ctx, x, fsmn_memory);

    return x;
}

void FSMNBlock::onload(const gguf_loader& loader, const std::string& prefix)
{
    LOAD_SUBMODULE_EX("linear.linear", linear);
    LOAD_SUBMODULE_EX("fsmn_block.conv_left", fsmn_block);
    LOAD_SUBMODULE_EX("affine.linear", affine);
}

ggml_tensor *FSMNBlock::build_cgraph(ggml_context *ctx, ggml_tensor *x) const
{
    return x;
}

void FeedForward::onload(const gguf_loader &loader, const std::string &prefix)
{
    LOAD_SUBMODULE_EX("ff.0.0", ff_0_0);
    LOAD_SUBMODULE_EX("ff.2", ff_2);
}

ggml_tensor *FeedForward::build_cgraph(ggml_context *ctx, ggml_tensor *x) const
{
    x = ff_0_0.build_cgraph(ctx, x);
    x = ggml_gelu_erf(ctx, x);
    x = ff_2.build_cgraph(ctx, x);
    return x;
}

void DiTBlock::onload(const gguf_loader &loader, const std::string &prefix)
{
    LOAD_SUBMODULE(attn_norm);
    LOAD_SUBMODULE(attn);
    LOAD_SUBMODULE(ff_norm);
    LOAD_SUBMODULE(ff);
}

ggml_tensor *DiTBlock::build_cgraph(ggml_context *ctx, ggml_tensor *x, ggml_tensor *time_emb, ggml_tensor *position_ids, int64_t cut_len) const 
{
    auto [norm, gate_msa, shift_mlp, scale_mlp, gate_mlp] = attn_norm.build_cgraph(ctx, x, time_emb);
    auto attn_output = attn.build_cgraph(ctx, norm, position_ids, cut_len);
    gate_msa = unsqueeze(ctx, gate_msa, 1);
    attn_output = ggml_mul(ctx, attn_output, gate_msa);
    if(cut_len > 0)
        x = ggml_view_3d(ctx, x, x->ne[0], x->ne[1] - cut_len, x->ne[2], x->nb[1], x->nb[2], x->nb[1] * cut_len);
    x = ggml_add(ctx, x, attn_output);

    auto ff_norm = this->ff_norm.build_cgraph(ctx, x);
    scale_mlp = ggml_scale_bias(ctx, ggml_cont(ctx, scale_mlp), 1.f, 1.f);
    ff_norm = ggml_mul(ctx, ff_norm, unsqueeze(ctx, scale_mlp, 1));
    ff_norm = ggml_add(ctx, ff_norm, unsqueeze(ctx, shift_mlp, 1));

    auto ff_output = ff.build_cgraph(ctx, ff_norm);
    ff_output = ggml_mul(ctx, ff_output, unsqueeze(ctx, gate_mlp, 1));
    x = ggml_add(ctx, x, ff_output);
    return x;
}

void DiT::onload(const gguf_loader &loader, const std::string &prefix)
{
    int heads;
    int depth; 

    LOAD_SUBMODULE(time_embed);
    LOAD_SUBMODULE(input_embed);

    LOAD_METADATA(heads);
    LOAD_METADATA(depth);
    LOAD_METADATA(mel_dim);

    transformer_blocks.resize(depth);
    for(int i = 0; i != depth; ++i)
    {
        auto &block = transformer_blocks[i];
        auto name = prefix + ".transformer_blocks." + std::to_string(i);
        block.onload(loader, name);
        block.attn.heads = heads;
        submodules[std::move(name)] = &transformer_blocks[i];
    }

    LOAD_SUBMODULE(norm_out);
    LOAD_SUBMODULE(proj_out);
}

ggml_tensor *DiT::build_cgraph(ggml_context *ctx, ggml_tensor *x, ggml_tensor *mu, 
                ggml_tensor *t, ggml_tensor *spks, ggml_tensor *cond, 
                int64_t cut_len, ggml_tensor * &ref_position_ids, 
                ggml_backend_op_capabilities_t capabilities) const
{    
    x = ggml_permute(ctx, x, 1, 0, 2, 3);

    t = time_embed.build_cgraph(ctx, t);
    x = input_embed.build_cgraph(ctx, x, cond, mu, spks, capabilities);

    ggml_tensor *position_ids = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, x->ne[1], x->ne[2]);
    ggml_set_input(position_ids);
    ref_position_ids = position_ids;
    position_ids = ggml_dup(ctx, position_ids);

    for (const auto& block : simple_span<const DiTBlock>(transformer_blocks.data(), transformer_blocks.size() - 1))
        x = block.build_cgraph(ctx, x, t, position_ids, 0);


    x = transformer_blocks.back().build_cgraph(ctx, x, t, position_ids, cut_len);
    x = norm_out.build_cgraph(ctx, x, t);

    auto output = proj_out.build_cgraph(ctx, x);
    return output;
}

void CausalConditionalCFM::onload(const gguf_loader &loader, const std::string &prefix)
{
    LOAD_SUBMODULE(estimator);
    LOAD_METADATA(inference_cfg_rate);

    for(int i = 0; i != 11; i++)
        t_span[i] = 1.f - std::cos(0.1f * 0.5f * 3.14159265358979323846f * i);
}


CausalConditionalCFM::DiTContext CausalConditionalCFM::prepare_context(ggml_context *ctx, ggml_tensor *mu, ggml_tensor *spks, ggml_tensor *cond) const 
{
    DiTContext dit_ctx{
        ggml_new_tensor_2d(
            ctx,
            GGML_TYPE_F32,
            mu->ne[1], 80
        ),

        ggml_pad(ctx, mu, 0, 0, 1, 0),
        ggml_pad(ctx, spks, 0, 0, 1, 0),
        ggml_pad(ctx, cond, 0, 0, 1, 0),
    };
    return dit_ctx;
}

std::array<float, 2> CausalConditionalCFM::get_t_and_dt(ggml_context *ctx, int step) const 
{
    GGML_ASSERT(step > 0 && step < t_span.size());

    auto t = t_span[0];

    for(int cur = 1; cur != step; ++cur)
        t += t_span[cur] - t_span[cur - 1];

    auto dt = t_span[step] - t_span[step - 1];

    return {t, dt};
}

ggml_tensor* CausalConditionalCFM::build_cgraph_one_step(ggml_context* ctx, 
                const DiTContext& dit_ctx, int step, ggml_backend_op_capabilities_t capabilities, 
                int64_t cut_len, ggml_tensor*& t_tensor, ggml_tensor*& position_ids) const

{
    auto x = dit_ctx.x;
    auto [t, dt] = get_t_and_dt(ctx, step);

    auto t_in = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 2);
    t_in = capabilities.fill ? ggml_fill_inplace(ctx, t_in, t) : ggml_scale_bias_inplace(ctx, t_in, 0.f, t);
    auto x_in = ggml_repeat_4d(ctx, x, x->ne[0], x->ne[1], 2, x->ne[3]);

    auto [dphi_dt, cfg_dphi_dt] = split_tensor<2>(
        ctx, estimator.build_cgraph(
            ctx,
            x_in,
            dit_ctx.mu_in, t_in,
            dit_ctx.spks_in,
            dit_ctx.cond_in,
            step == t_span.size() - 1 ? cut_len : 0,
            position_ids,
            capabilities
        ), 2
    );

    dphi_dt = ggml_permute(ctx, dphi_dt, 1, 0, 2, 3);
    cfg_dphi_dt = ggml_permute(ctx, cfg_dphi_dt, 1, 0, 2, 3);
    dphi_dt = ggml_cont(ctx, dphi_dt);
    cfg_dphi_dt = ggml_cont(ctx, cfg_dphi_dt);
    cfg_dphi_dt = ggml_scale(ctx, cfg_dphi_dt, inference_cfg_rate);
    dphi_dt = ggml_scale(ctx, dphi_dt, 1.f + inference_cfg_rate);
    dphi_dt = ggml_sub(ctx, dphi_dt, cfg_dphi_dt);

    if(step == t_span.size() - 1 && cut_len > 0)
        x = ggml_view_3d(ctx, x, x->ne[0] - cut_len, x->ne[1], x->ne[2], x->nb[1], x->nb[2], x->nb[0] * cut_len);

    x = ggml_add(ctx, x, ggml_scale(ctx, dphi_dt, dt));
    t_tensor = t_in;
    return x;
}


void PreLookaheadLayer::onload(const gguf_loader &loader, const std::string &prefix)
{
    LOAD_METADATA(pre_lookahead_len);

    LOAD_SUBMODULE(conv1);
    LOAD_SUBMODULE(conv2);
}

ggml_tensor *PreLookaheadLayer::build_cgraph(ggml_context *ctx, ggml_tensor *inputs) const
{
    auto outputs = ggml_permute(ctx, inputs, 1, 0, 2, 3);
    outputs = ggml_cont(ctx, outputs);
    outputs = ggml_pad(ctx, outputs, pre_lookahead_len, 0, 0, 0);
    outputs = conv1.build_cgraph(ctx, outputs, 1, 0, 1, 1, {});
    outputs = ggml_leaky_relu(ctx, outputs, 0.01f, false);

    outputs = ggml_pad_ext(ctx, outputs, static_cast<int>(conv2.weight->ne[0] - 1), 0, 0, 0, 0, 0, 0, 0);
    outputs = conv2.build_cgraph(ctx, outputs, 1, 0, 1, 1, {});
    outputs = ggml_permute(ctx, outputs, 1, 0, 2, 3);

    outputs = ggml_cont(ctx, outputs);
    outputs = ggml_add(ctx, outputs, inputs);

    return outputs;
}

std::array<ggml_tensor *, 2>SineGen2::build_cgraph(ggml_context *ctx, ggml_tensor *f0, 
                           int harmonic_num, int sampling_rate, int upsample_scale, float sine_amp,
                           int voiced_threshold, float noise_std) const
{
    auto fn = ggml_arange(ctx, 1.f, static_cast<float>(harmonic_num + 2), 1.f);
    fn = ggml_mul(ctx, ggml_repeat_4d(ctx, f0, harmonic_num + 1, f0->ne[1], f0->ne[2], 1), fn);

    // _f02sine
    auto rad_values = ggml_scale(ctx, fn, 1.f / sampling_rate);
    ggml_tensor *sine_waves;
    {
        rad_values = ggml_sub(ctx, rad_values, ggml_floor(ctx, rad_values));

        rad_values = view_tensor(ctx, rad_values, 
                ggml_acc(ctx, rad_values, rand_ini, rad_values->nb[1], rad_values->nb[2], rad_values->nb[3], 0));

        rad_values = ggml_permute(ctx, rad_values, 1, 0, 2, 3);
        rad_values = ggml_interpolate(ctx, rad_values, rad_values->ne[0] / upsample_scale, rad_values->ne[1], rad_values->ne[2], 1, GGML_SCALE_MODE_BILINEAR);

    
        auto phase = ggml_cumsum(ctx, rad_values);
        phase = ggml_scale(ctx, phase, 2.0f * 3.14159265358979323846f * upsample_scale);
        phase = ggml_permute(ctx, phase, 1, 0, 2, 3);
        phase = ggml_interpolate(ctx, phase, phase->ne[0], phase->ne[1] * upsample_scale, phase->ne[2], 1, GGML_SCALE_MODE_NEAREST);
        auto sines = ggml_sin(ctx, phase);
        sine_waves = ggml_scale(ctx, sines, sine_amp);
    }

    // _f02uv
    ggml_tensor *uv;
    {
        constexpr float eps = 1e-7f;
        auto x = ggml_scale_bias(ctx, f0, 1.f, static_cast<float>(-voiced_threshold));
        auto pos = ggml_relu(ctx, x);
        auto denom = ggml_scale_bias(ctx, ggml_abs(ctx, x), 1.f, eps);
        uv = ggml_div(ctx, pos, denom);
    }

    auto noise_amp = ggml_scale_bias(ctx, uv, noise_std - sine_amp / 3, sine_amp / 3);
    auto noise_orig = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, sine_waves->ne[0], sine_waves->ne[1]);
    auto noise = ggml_mul(ctx, noise_orig, noise_amp);

    sine_waves = ggml_mul(ctx, sine_waves, uv);
    sine_waves = ggml_add(ctx, sine_waves, noise);

    return {sine_waves, noise_orig };
}

void SourceModuleHnNSF::onload(const gguf_loader &loader, const std::string &prefix)
{
    LOAD_SUBMODULE(l_linear);
}

std::array<ggml_tensor*, 2> SourceModuleHnNSF::build_cgraph(ggml_context* ctx, ggml_tensor* x, int harmonic_num, int sampling_rate, int upsample_scale, float sine_amp, int voiced_threshold, float noise_std) const
{
    auto [sine_wavs, noise] = l_sin_gen.build_cgraph(ctx, x, harmonic_num, sampling_rate, upsample_scale, sine_amp, voiced_threshold, noise_std);
    auto sine_merge = l_linear.build_cgraph(ctx, sine_wavs);
    sine_merge = ggml_tanh(ctx, sine_merge);
    return {sine_merge, noise};
}

ggml_tensor *SinusPositionEmbedding::build_cgraph(ggml_context *ctx, ggml_tensor *x) const
{
    x = unsqueeze(ctx, x, 0);
    x = ggml_repeat_4d(ctx, x, emb->ne[0], x->ne[1], x->ne[2], x->ne[3]);
    auto embedding = ggml_mul(ctx, x, emb); 
    auto sin_emb = ggml_sin(ctx, embedding);
    auto cos_emb = ggml_cos(ctx, embedding);
    return concat_tensors(ctx, std::array{sin_emb, cos_emb}, 0);
}


void EncoderLayerSANM::onload(const gguf_loader& loader, const std::string& prefix)
{
    LOAD_SUBMODULE(self_attn);

    LOAD_SUBMODULE(feed_forward);

    LOAD_SUBMODULE(norm1);
    LOAD_SUBMODULE(norm2);
}

ggml_tensor *EncoderLayerSANM::build_cgraph(ggml_context *ctx, ggml_tensor *x, int n_hidden_state, int n_head, int fsmn_kernel_size, int flash_attn) const
{
    ggml_tensor *residual = nullptr;

    if(norm1.weight->ne[0] == norm2.weight->ne[0])
    {
        residual = ggml_cpy(ctx, x, ggml_new_tensor_3d(ctx, x->type, x->ne[0], x->ne[1], x->ne[2]));
    }

    x = norm1.build_cgraph(ctx, x);

    x = self_attn.build_cgraph(ctx, x, n_hidden_state, n_head, fsmn_kernel_size, flash_attn);

    if(norm1.weight->ne[0] == norm2.weight->ne[0])
    {
        x = ggml_add(ctx, x, residual);
    }

    residual = ggml_cpy(
            ctx, x,
            ggml_new_tensor_3d(ctx, GGML_TYPE_F32, x->ne[0], x->ne[1], x->ne[2]));

    x = norm2.build_cgraph(ctx, x);
    

    x = feed_forward.build_cgraph(ctx, x);

    // residual after position wise feed forward
    x = ggml_add(ctx, x, residual);
    return x;
}

void Qwen2MLP::onload(const gguf_loader &loader, const std::string &prefix)
{
    LOAD_SUBMODULE(gate_proj);
    LOAD_SUBMODULE(up_proj);
    LOAD_SUBMODULE(down_proj);
}

ggml_tensor *Qwen2MLP::build_cgraph(ggml_context *ctx, ggml_tensor *x) const
{
    auto gate = gate_proj.build_cgraph(ctx, x);
    auto up_states = up_proj.build_cgraph(ctx, x);

    up_states = ggml_swiglu_split(ctx, gate, up_states);
    return down_proj.build_cgraph(ctx, up_states);
}

void Qwen2RMSNorm::onload(const gguf_loader &loader, const std::string &prefix)
{
    LOAD_TENSOR(weight);
}

ggml_tensor *Qwen2RMSNorm::build_cgraph(ggml_context *ctx, ggml_tensor *hidden_states, float variance_epsilon) const
{
    hidden_states = ggml_rms_norm(ctx, hidden_states, variance_epsilon);
    hidden_states = ggml_mul(ctx, hidden_states, weight);
    return hidden_states;
}

void Qwen2Attention::onload(const gguf_loader &loader, const std::string &prefix)
{
    LOAD_SUBMODULE(q_proj);
    LOAD_SUBMODULE(k_proj);
    LOAD_SUBMODULE(v_proj);
    LOAD_SUBMODULE(o_proj);
}

void Qwen2DecoderLayer::onload(const gguf_loader &loader, const std::string &prefix)
{
    LOAD_SUBMODULE(self_attn);
    LOAD_SUBMODULE(mlp);
    LOAD_SUBMODULE(input_layernorm);
    LOAD_SUBMODULE(post_attention_layernorm);
}


void SenseVoiceEncoderSmall::onload(const gguf_loader &loader, std::string prefix)
{
    prefix="";
    LOAD_SUBMODULE(embed);

    //to do 
    prefix="encoder";
    LOAD_METADATA(output_size);
    LOAD_METADATA(linear_units);
    LOAD_METADATA(attention_heads);
    LOAD_METADATA(num_blocks);
    LOAD_METADATA(tp_blocks);

    LOAD_SUBMODULE(after_norm);
    LOAD_SUBMODULE(tp_norm);

    //LOG_DEBUG("SenseVoiceEncoderSmall output_size %d linear_units %d attention_heads %d num_blocks %d tp_blocks %d ", output_size, linear_units, attention_heads, num_blocks, tp_blocks);

    prefix = "encoder.encoders0";
    for(int i = 0; i < 1; i++)
    {
        LOAD_SUBMODULE_EX(std::to_string(i).c_str(), encoders0);
    }

    prefix = "encoder.encoders";
    encoders.resize(num_blocks - 1);
    for(int i = 0; i < num_blocks - 1; i++)
    {
        LOAD_SUBMODULE_EX(std::to_string(i).c_str(), encoders[i]);
    }

    prefix = "encoder.tp_encoders";
    tp_encoders.resize(tp_blocks);
    for(int i = 0; i < tp_blocks; i++)
    {
        LOAD_SUBMODULE_EX(std::to_string(i).c_str(), tp_encoders[i]);
    }
}

ggml_tensor *SenseVoiceEncoderSmall::build_cgraph(ggml_context *ctx, ggml_tensor *x, int fsmn_kernel_size, int flash_attn) const
{
    // [x] 1. sinusoidal position
    // [x] 2. encoders0
    // [x] 3. encoders
    // [x] 4. tp_encoders
    // [x] 5. tp_norm
    x = embed.build_cgraph(ctx, x, output_size);

    x = encoders0.build_cgraph(ctx, x, output_size, attention_heads, fsmn_kernel_size, flash_attn);

    for(int i = 0; i < num_blocks - 1; i++)
    {
        x = encoders[i].build_cgraph(ctx, x, output_size, attention_heads, fsmn_kernel_size, flash_attn);
    }
    x = after_norm.build_cgraph(ctx, x);

    for(int i = 0; i < tp_blocks; i++)
    {
        x = tp_encoders[i].build_cgraph(ctx, x, output_size, attention_heads, fsmn_kernel_size, flash_attn);
    }
    x = tp_norm.build_cgraph(ctx, x);

    return x;
}

void CausalMaskedDiffWithDiT::onload(const gguf_loader &loader, const std::string &prefix)
{
    LOAD_METADATA(token_mel_ratio);

    LOAD_TENSOR_EX("input_embedding.weight", input_embedding);
    LOAD_SUBMODULE(spk_embed_affine_layer);
    LOAD_SUBMODULE(pre_lookahead_layer);
    LOAD_SUBMODULE(decoder);
}

CausalMaskedDiffWithDiT::EncodeResult CausalMaskedDiffWithDiT::build_cgraph_encode(
                ggml_context *ctx, ggml_tensor *token, ggml_tensor *prompt_token, 
                ggml_tensor *prompt_feat, ggml_tensor *embedding, 
                ggml_backend_op_capabilities_t capabilities) const
{
    embedding = ggml_l2_norm(ctx, embedding, 1e-6f);
    embedding = spk_embed_affine_layer.build_cgraph(ctx, embedding);

    token = concat_tensors(ctx, std::array { prompt_token, token }, 0, capabilities);
    token = ggml_get_rows(ctx, input_embedding, token);

    auto h = pre_lookahead_layer.build_cgraph(ctx, token);
    h = unsqueeze(ctx, h, 1);
    h = ggml_repeat_4d(ctx, h, h->ne[0], token_mel_ratio, h->ne[2], h->ne[3]);
    h = ggml_reshape_3d(ctx, h, h->ne[0], h->ne[1] * h->ne[2], h->ne[3]);

    const auto mel_len1 = prompt_feat->ne[1];
    const auto mel_len2 = h->ne[1] - mel_len1;
    auto conds = ggml_pad(ctx, prompt_feat, 0, static_cast<int>(mel_len2), 0, 0);

    return EncodeResult{
        h, embedding, conds, mel_len1,
    };
}

void CausalHiFTGenerator::onload(const gguf_loader& loader, const std::string& prefix)
{
    LOAD_SUBMODULE(f0_predictor);
    LOAD_SUBMODULE(m_source);
    LOAD_SUBMODULE(conv_pre);
    conv_pre.causal_type = CausalConv1d::causal_type_t::right;

    int num_kernels;
    GGML_ASSERT(loader.get_metadata("sample_rate", reinterpret_cast<uint32_t&>(sampling_rate)));
    LOAD_METADATA(num_kernels);
    LOAD_METADATA(nb_harmonics);
    LOAD_METADATA(nsf_alpha);
    LOAD_METADATA(nsf_voiced_threshold);
    LOAD_METADATA(nsf_sigma);
    LOAD_METADATA(lrelu_slope);
    LOAD_METADATA(audio_limit);
    GGML_ASSERT(loader.get_metadata(prefix, "istft_params.n_fft", nfft));
    GGML_ASSERT(loader.get_metadata(prefix, "istft_params.hop_len", hop_len));

    int64_t id;
    GGML_ASSERT(loader.find_metadata_key(combine_prefix(prefix, "upsample_rates").c_str(), id));
    GGML_ASSERT(gguf_get_arr_type(loader.gguf_ctx, id) == GGUF_TYPE_INT32);

    auto layers = gguf_get_arr_n(loader.gguf_ctx, id);
    auto upsample_rates = reinterpret_cast<const int*>(gguf_get_arr_data(loader.gguf_ctx, id));

    ups.resize(layers);
    scale_factor = hop_len;
    for (size_t i = 0; i != layers; ++i)
    {
        auto& up = ups[i];
        int upsample_rate = upsample_rates[i];
        scale_factor *= upsample_rate;
        up.s = upsample_rate;
        LOAD_SUBMODULE_EX(("ups." + std::to_string(i)).c_str(), up);
    }
    
    source_downs.resize(layers);
    source_resblocks.resize(layers);
    for (size_t i = 0; i != layers - 1; ++i)
    {
        auto down = new CausalConv1dDownSample;
        down->s = 1;
        for (size_t j = layers - 1; j > i; --j)
            down->s *= upsample_rates[j];
        source_downs[i].reset(down);

        LOAD_SUBMODULE_EX(("source_downs." + std::to_string(i)).c_str(), *down);
        LOAD_SUBMODULE_EX(("source_resblocks." + std::to_string(i)).c_str(), source_resblocks[i]);
    }
    // Final block.
    {
        const auto layer_idx = layers - 1;
        auto conv = new CausalConv1d;
        conv->d = 1;
        conv->causal_type = CausalConv1d::causal_type_t::left;
        source_downs[layer_idx].reset(conv);

        LOAD_SUBMODULE_EX(("source_downs." + std::to_string(layer_idx)).c_str(), *conv);
        LOAD_SUBMODULE_EX(("source_resblocks." + std::to_string(layer_idx)).c_str(), source_resblocks[layer_idx]);
    }

    layers *= num_kernels;
    resblocks.resize(layers);
    for (size_t i = 0; i != layers; ++i)
        LOAD_SUBMODULE_EX(("resblocks." + std::to_string(i)).c_str(), resblocks[i]);

    LOAD_SUBMODULE(conv_post);
    conv_post.causal_type = CausalConv1d::causal_type_t::left;
}

std::array<ggml_tensor*, 2> CausalHiFTGenerator::build_cgraph(ggml_context* ctx, ggml_tensor* speech_feat) const
{
    auto f0 = f0_predictor.build_cgraph(ctx, speech_feat);
    auto s_input = ggml_permute(ctx, f0, 1, 0, 2, 3);
    s_input = ggml_interpolate(ctx, s_input, s_input->ne[0], s_input->ne[1] * scale_factor, s_input->ne[2], 1, GGML_SCALE_MODE_NEAREST);
    auto [s, noise] = m_source.build_cgraph(ctx, s_input, nb_harmonics, sampling_rate, scale_factor, nsf_alpha, nsf_voiced_threshold, nsf_sigma);
    s = ggml_permute(ctx, s, 1, 0, 2, 3);

    // decode
    ggml_tensor* x;
    {
        x = conv_pre.build_cgraph(ctx, speech_feat);

        auto s_stft = ggml_stft(ctx, s, window, hop_len, true, fctx.get());
        s_stft = ggml_cont(ctx, s_stft);
        s_stft = ggml_reshape_2d(ctx, s_stft, s_stft->ne[0], s_stft->ne[1] * 2);

        const auto num_upsamples = ups.size();
        const auto num_kernels = resblocks.size() / num_upsamples;
        for (size_t i = 0; i != num_upsamples; ++i)
        {
            x = ggml_leaky_relu(ctx, x, lrelu_slope, false);
            x = ups[i].build_cgraph(ctx, x);

            if (i == num_upsamples - 1)
                x = ggml_pad_reflect_1d(ctx, x, 1, 0);

            auto si = source_downs[i]->build_cgraph(ctx, s_stft);
            si = source_resblocks[i].build_cgraph(ctx, si);
            x = ggml_add(ctx, x, si);

            auto xs = resblocks[i * num_kernels].build_cgraph(ctx, x);
            for (size_t j = 1; j != num_kernels; ++j)
                xs = ggml_add(ctx, xs,
                    resblocks[i * num_kernels + j].build_cgraph(ctx, x));
            x = ggml_scale(ctx, xs, 1.f / num_kernels);
        }
        x = ggml_leaky_relu(ctx, x, 0.01f, false);
        x = conv_post.build_cgraph(ctx, x);

        auto magnitude = ggml_view_3d(ctx, x, x->ne[0], nfft / 2 + 1, x->ne[2], x->nb[1], x->nb[2], 0);
        magnitude = ggml_exp(ctx, magnitude);
        auto phase = ggml_view_3d(ctx, x, x->ne[0], nfft / 2 + 1, x->ne[2], x->nb[1], x->nb[2], (nfft / 2 + 1) * x->nb[1]);
        phase = ggml_sin(ctx, phase);

        // istft
        {
            magnitude = ggml_clamp(ctx, magnitude, 0.f, 1e2f);
            auto real = ggml_mul(ctx, magnitude,
                ggml_cos(ctx, phase));
            auto imag = ggml_mul(ctx, magnitude,
                ggml_sin(ctx, phase));

            x = ggml_istft(ctx, real, imag, window, hop_len, true, ictx.get());
            x = ggml_clamp(ctx, x, -audio_limit, audio_limit);
            ggml_set_cpu(x);
            return { x, noise };
        }
    }
}

#if 0

std::array<ggml_tensor *, 2>CausalHiFTGenerator::build_cgraph(ggml_context *ctx, ggml_tensor *speech_feat) const
{
    auto f0 = f0_predictor.build_cgraph(ctx, speech_feat);
    auto s_input = ggml_permute(ctx, f0, 1, 0, 2, 3);
    s_input = ggml_interpolate(ctx, s_input, s_input->ne[0], s_input->ne[1] * scale_factor, s_input->ne[2], 1, GGML_SCALE_MODE_NEAREST);
    auto [s, noise] = m_source.build_cgraph(ctx, s_input, nb_harmonics, sampling_rate, scale_factor,
            nsf_alpha, nsf_voiced_threshold, nsf_sigma);
    s = ggml_permute(ctx, s, 1, 0, 2, 3);

    /* decode */
    ggml_tensor *x;
    {
        x = conv_pre.build_cgraph(ctx, speech_feat);

		auto s_stft = ggml_stft(ctx, s, window, hop_len, true, fctx.get());
		s_stft = ggml_cont(ctx, s_stft);
		s_stft = ggml_reshape_2d(ctx, s_stft, s_stft->ne[0], s_stft->ne[1] * 2);

        const auto num_upsamples = ups.size();
        const auto num_kernels = resblocks.size() / num_upsamples;
        for(size_t i = 0; i != num_upsamples; ++i)
        {
            x = ggml_leaky_relu(ctx, x, lrelu_slope, false);
            x = ups[i].build_cgraph(ctx, x);

            if(i == num_upsamples -1)
                x = ggml_pad_reflect_1d(ctx, x, 1, 0);

			auto si = source_downs[i]->build_cgraph(ctx, s_stft);
			si = source_resblocks[i].build_cgraph(ctx, si);
			x = ggml_add(ctx, x, si);

            auto xs = resblocks[i * num_kernels].build_cgraph(ctx, x);
            for(size_t j = 1; j != num_kernels; ++j)
            {
                xs = ggml_add(ctx, xs,
                        resblocks[i * num_kernels + j].build_cgraph(ctx, x));
            }
            x = ggml_scale(ctx, xs, 1.f / num_kernels);
        }
        x = ggml_leaky_relu(ctx, x, 0.01f, false);
        x = conv_post.build_cgraph(ctx, x);

        auto magnitude = ggml_view_3d(ctx, x, x->ne[0], nfft / 2 + 1, x->ne[2], x->nb[1], x->nb[2], 0);
        magnitude = ggml_exp(ctx, magnitude);
        auto phase = ggml_view_3d(ctx, x, x->ne[0], nfft / 2 + 1, x->ne[2], x->nb[1], x->nb[2], (nfft / 2 + 1) * x->nb[1]);
        phase = ggml_sin(ctx, phase);
        /* istft */
        {
            magnitude = ggml_clamp(ctx, magnitude, 0.f, 1e2f);
            auto real = ggml_mul(ctx, magnitude, ggml_cos(ctx, phase));
            auto imag = ggml_mul(ctx, magnitude, ggml_sin(ctx, phase));

            x = ggml_istft(ctx, real, imag, window, hop_len, true, ictx.get());
            x = ggml_clamp(ctx, x, -audio_limit, audio_limit);
            ggml_set_cpu(x);
            return {x, noise};
        }
    }
}
#endif


void CosyVoice3LM::onload(const gguf_loader &loader, const std::string &prefix)
{
    int num_hidden_layers;

    LOAD_TENSOR_EX("embed_tokens.weight", embed_tokens_weight);
    LOAD_TENSOR_EX("speech_embedding.weight", speech_embedding_weight);
    LOAD_SUBMODULE(norm);
    LOAD_SUBMODULE(llm_decoder);

    LOAD_METADATA(num_hidden_layers);
    LOAD_METADATA(num_attention_heads);
    LOAD_METADATA(num_key_value_heads);
    LOAD_METADATA(rms_norm_eps);
    LOAD_METADATA(rope_theta);
    LOAD_METADATA(sos_token_id);
    LOAD_METADATA(task_token_id);

    GGML_ASSERT(embed_tokens_weight->ne[0] == speech_embedding_weight->ne[0]);
    layers.resize(num_hidden_layers);

    auto layers_prefix = combine_prefix(prefix, "layers");
    for(int i = 0; i != num_hidden_layers; ++i)
    {
        auto &layer = layers[i];
        auto name = layers_prefix + "." + std::to_string(i);
        LOAD_SUBMODULE_EX(name.c_str(), layer);
    }
}


