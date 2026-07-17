#include "fft.h"
#include "ggml_fft.h"
#include "ggml_cpu_flag.h"

#include <cstring>
#include <atomic>
#include <math.h>

#if 0
#if defined(__x86_64__) || defined(_M_X64)
#include <immintrin.h>
#elif defined(__aarch64__) || defined(_M_ARM64)
#define SIMDE_ENABLE_NATIVE_ALIASES
#include <simde/x86/avx2.h>
#include <simde/x86/fma.h>
#else
#error "ggml_fft requires x86_64 SIMD intrinsics or SIMDe on ARM64; unsupported architecture"
#endif


struct fft_context
{
    fft_context(int nfft);
    ~fft_context() = default;

    int factors[63];
    int max_factor_value;
    int nfft;
    std::atomic_uint workers;
    std::atomic_int64_t idx;
    std::unique_ptr<float[]> twiddles_r;
    float* twiddles_i;
};

static
void kf_work(
    float* Foutr,
    float* Fouti,
    const float* f,
    int fstride,
    int in_stride,
    int* factors,
    float* buffer_scratchr,
    float* buffer_scratchi,
    fft_context& ctx
)
{
    int Fout_beg = 0;
    const int p = *factors++; /* the radix  */
    const int m = *factors++; /* stage's fft length/p */
    const int Fout_end = p * m;

    if (m == 1) {
        do {
            Foutr[Fout_beg] = *f;
            f += fstride * in_stride;
        } while (++Fout_beg != Fout_end);
    }
    else {
        do {
            // recursive call:
            // DFT of size m*p performed by doing
            // p instances of smaller DFTs of size m,
            // each one takes a decimated version of the input
            kf_work(Foutr + Fout_beg, Fouti + Fout_beg, f, fstride * p, in_stride, factors, buffer_scratchr, buffer_scratchi, ctx);
            f += fstride * in_stride;
        } while ((Fout_beg += m) != Fout_end);
    }

    // recombine the p smaller DFTs
    switch (p) {
    case 2: kf_bfly2(Foutr, Fouti, fstride, ctx.twiddles_r.get(), ctx.twiddles_i, m); break;
    case 3: kf_bfly3(Foutr, Fouti, fstride, ctx.twiddles_r.get(), ctx.twiddles_i, m); break;
    case 4: kf_bfly4(Foutr, Fouti, fstride, ctx.twiddles_r.get(), ctx.twiddles_i, m); break;
    case 5: kf_bfly5(Foutr, Fouti, fstride, ctx.twiddles_r.get(), ctx.twiddles_i, m); break;
    default: kf_bfly_generic(Foutr, Fouti, fstride, ctx.twiddles_r.get(), ctx.twiddles_i, ctx.nfft, m, p, buffer_scratchr, buffer_scratchi);
    }
}


fft_context::fft_context(int nfft) : nfft(nfft), idx(0), workers(0), twiddles_r(new float[nfft * 2])
{
    twiddles_i = twiddles_r.get() + nfft;

    int i = 0;
    constexpr float pi = 3.1415926535897932f;
    const float factor = -2 * pi / nfft;
    // Note: SVML intrinsics (_mm256_cos_ps, etc.) require VS2019+ or Intel compiler.
    // Use scalar cosf/sinf for VS2017 compatibility.
    for (; i < nfft; ++i)
    {
        float phase = factor * i;
        twiddles_r[i] = cosf(phase);
        twiddles_i[i] = sinf(phase);
    }

    int p = 4;
    auto n = nfft;
    double floor_sqrt = floor(sqrt(n));
    auto facbuf = factors;
    max_factor_value = 0;

    /*factor out powers of 4, powers of 2, then any remaining primes */
    do {
        while (n % p) {
            switch (p) {
            case 4: p = 2; break;
            case 2: p = 3; break;
            default: p += 2; break;
            }
            if (p > floor_sqrt)
                p = n;          /* no more factors, skip to end */
        }
        n /= p;
        *facbuf++ = p;
        *facbuf++ = n;
        if (p > max_factor_value)
            max_factor_value = p;
        if (n > max_factor_value)
            max_factor_value = n;
    } while (n > 1);
}

static void ggml_fft_op(struct ggml_tensor* dst, int ith, int nth, fft_context* ctx)
{
    auto buffer_scratchr = reinterpret_cast<float*>(alloca(sizeof(float) * ctx->max_factor_value * 2));
    auto buffer_scratchi = buffer_scratchr + ctx->max_factor_value;
    auto const signal = dst->src[0];
    auto const data = reinterpret_cast<char*>(signal->data);
    const auto foutr = reinterpret_cast<char*>(dst->data);
    const auto fouti = foutr + dst->nb[2];
    const auto dst_nb1 = dst->nb[1];
    const auto signal_nb1 = signal->nb[1];
    const auto signal_ne1 = signal->ne[1];

    ++ctx->workers;

    for (int64_t i = ctx->idx++; i < signal_ne1; i = ctx->idx++)
    {
        const auto cur_offset = dst_nb1 * i;
        const auto cur_fouti = reinterpret_cast<float*>(fouti + cur_offset);
        memset(cur_fouti, 0, dst_nb1);
        kf_work(reinterpret_cast<float*>(foutr + cur_offset),
            cur_fouti,
            reinterpret_cast<float*>(data + signal_nb1 * i),
            1, 1, ctx->factors,
            buffer_scratchr, buffer_scratchi, *ctx);
    }

    if (--ctx->workers == 0) ctx->idx = 0;
}

void delete_fft_context(fft_context* ctx)
{
    delete ctx;
}

fft_context_ptr create_fft_context(int nfft)
{
    return fft_context_ptr(new fft_context(nfft));
}

void fft(const float* signal, float* buffer, fft_context& ctx)
{
    auto nfft = ctx.nfft;
    auto foutr = reinterpret_cast<float*>(alloca(sizeof(float) * nfft * 2 + sizeof(float) * ctx.max_factor_value * 2));
    auto fouti = foutr + nfft;
    auto buffer_scratchr = fouti + nfft;
    auto buffer_scratchi = buffer_scratchr + ctx.max_factor_value;
    memset(fouti, 0, sizeof(float) * nfft);

    kf_work(foutr, fouti, signal, 1, 1, ctx.factors, buffer_scratchr, buffer_scratchi, ctx);
    int i = 0;
    while (i + 7 < nfft)
    {
        __m256 tr = _mm256_loadu_ps(foutr + i);
        __m256 ti = _mm256_loadu_ps(fouti + i);
        auto abs = _mm256_sqrt_ps(_mm256_fmadd_ps(tr, tr, _mm256_mul_ps(ti, ti)));
        _mm256_storeu_ps(buffer + i, abs);
        i += 8;
    }
    while (i + 3 < nfft)
    {
        __m128 tr = _mm_loadu_ps(foutr + i);
        __m128 ti = _mm_loadu_ps(fouti + i);
        auto abs = _mm_sqrt_ps(_mm_fmadd_ps(tr, tr, _mm_mul_ps(ti, ti)));
        _mm_storeu_ps(buffer + i, abs);
        i += 4;
    }
    for (; i != nfft; ++i)
        buffer[i] = sqrtf(foutr[i] * foutr[i] + fouti[i] * fouti[i]);
}

ggml_tensor* ggml_fft(ggml_context* ctx, ggml_tensor* a, fft_context* fctx)
{
    GGML_ASSERT(ggml_is_matrix(a));
    GGML_ASSERT(a->ne[0] == fctx->nfft);
    GGML_ASSERT(ggml_is_contiguous(a));

    return ggml_custom_4d(ctx, a->type, fctx->nfft, a->ne[1], 2, 1, &a, 1, reinterpret_cast<ggml_custom_op_t>(ggml_fft_op), GGML_N_TASKS_MAX, fctx);
}

ggml_tensor* ggml_stft(ggml_context* ctx, ggml_tensor* a, ggml_tensor* b, int hop_len, bool center, fft_context* fctx)
{
    GGML_ASSERT(ggml_is_vector(a));
    GGML_ASSERT(ggml_is_vector(b));
    GGML_ASSERT(ggml_is_contiguous(a));

    if (center)
    {
        const auto pad_amount = fctx->nfft / 2;
        a = ggml_pad_reflect_1d(ctx, a, pad_amount, pad_amount);
    }
    else
        a = ggml_dup(ctx, a);

    ggml_set_cpu(a);

    const auto win_size = b->ne[0];
    const auto n_frames = (a->ne[0] - win_size) / hop_len + 1;

    a = ggml_dup_inplace(ctx, a);
    a->ne[0] = win_size;
    a->ne[1] = n_frames;
    a->nb[1] = hop_len * a->nb[0];
    a->op = GGML_OP_VIEW;
    ggml_set_cpu(a);

    b = ggml_repeat(ctx, b, a);
    ggml_set_cpu(b);
    a = ggml_mul(ctx, a, b);
    ggml_set_cpu(a);

    auto result = ggml_fft(ctx, a, fctx);
    result = ggml_view_3d(ctx, result, fctx->nfft / 2 + 1, n_frames, 2, result->nb[1], result->nb[2], 0);
    return ggml_permute(ctx, result, 1, 0, 2, 3);
}

struct istft_context
{
    int nfft;
    ggml_tensor* idft_real;
    ggml_tensor* idft_imag;
};

void delete_istft_context(istft_context* ctx)
{
    delete ctx;
}

istft_context_ptr create_istft_context(int nfft, ggml_context* ctx, std::function<void(ggml_tensor*, void*, size_t)> set_data)
{
    const int f = nfft / 2 + 1;
    const int stride = f * nfft;
    std::unique_ptr<float[]> idft_real(new float[stride * 2]);
    auto idft_imag = idft_real.get() + stride;
    for (int y = 0; y != f; ++y)
        for (int x = 0; x != nfft; ++x)
        {
            const float phase = 2.0f * 3.14159265358979323846f * x * y / nfft;
            const int offset = x * f + y;
            idft_real[offset] = std::cos(phase);
            idft_imag[offset] = phase;
        }

    for (int x = 0, y = f - 1; x != nfft; ++x)
    {
        const auto offset1 = x * f;
        const auto offset2 = offset1 + y;
        idft_real[offset1] /= nfft;
        idft_real[offset2] /= nfft;
        idft_imag[offset2] = 0.0f;
    }

    for (int y = 1, endy = f - 1; y != endy; ++y)
        for (int x = 0; x != nfft; ++x)
        {
            const int idx = x * f + y;
            auto& real = idft_real[idx];
            auto& imag = idft_imag[idx];

            real = 2.0f * real / nfft;
            imag = -2.0f * std::sin(imag) / nfft;
        }

    auto ictx = new istft_context;
    ictx->nfft = nfft;
    ictx->idft_real = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, f, nfft);
    ictx->idft_imag = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, f, nfft);
    set_data(ictx->idft_real, idft_real.get(), sizeof(float) * stride);
    set_data(ictx->idft_imag, idft_imag, sizeof(float) * stride);
    return istft_context_ptr(ictx);
}

static void ggml_istft_finalize_op(struct ggml_tensor* dst, int ith, int nth, void* userdata)
{
    auto idx = dst->src[0];
    auto idx_data = reinterpret_cast<int32_t*>(idx->data);
    auto frames = reinterpret_cast<float*>(dst->src[1]->data);
    auto w_sq = reinterpret_cast<float*>(dst->src[2]->data);
    auto dst_data = reinterpret_cast<float*>(dst->data);
    auto denom_data = reinterpret_cast<float*>(dst->src[3]->data);

    const auto offset = userdata ? reinterpret_cast<intptr_t>(userdata) : 0;
    const auto out_len = dst->ne[0];
    auto win_len = dst->src[2]->ne[0];

    const auto begin = (out_len * ith) / nth;
    const auto end = (out_len * (ith + 1)) / nth;

    memset(dst_data + begin, 0, (end - begin) * sizeof(float));
    memset(denom_data + begin, 0, (end - begin) * sizeof(float));
    const auto n = ggml_nelements(idx);
    for (int64_t i = 0; i != n; ++i) {
        const auto cur = idx_data[i] - offset;
        if (cur < begin || cur >= end)
            continue;

        dst_data[cur] += frames[i];
        denom_data[cur] += w_sq[i % win_len];
    }

    constexpr float eps = 1e-8f;
    int64_t i = begin;
    for (const __m256 eps_v = _mm256_set1_ps(eps); i + 7 < end; i += 8) {
        auto y_ptr = dst_data + i;
        __m256 denom = _mm256_loadu_ps(denom_data + i);
        __m256 y = _mm256_loadu_ps(y_ptr);
        denom = _mm256_max_ps(denom, eps_v);
        y = _mm256_div_ps(y, denom);
        _mm256_storeu_ps(y_ptr, y);
    }

    for (const __m128 eps_v = _mm_set_ps1(eps); i + 3 < end; i += 4) {
        auto y_ptr = dst_data + i;
        __m128 denom = _mm_loadu_ps(denom_data + i);
        __m128 y = _mm_loadu_ps(y_ptr);
        denom = _mm_max_ps(denom, eps_v);
        y = _mm_div_ps(y, denom);
        _mm_storeu_ps(y_ptr, y);
    }

    for (; i < end; ++i) {
        auto denom = denom_data[i];
        denom = std::max(denom, eps);
        dst_data[i] /= denom;
    }
}

ggml_tensor* ggml_istft(ggml_context* ctx, ggml_tensor* a, ggml_tensor* b, ggml_tensor* c, int hop_len, bool center, istft_context* ictx)
{
    GGML_ASSERT(ggml_are_same_shape(a, b));
    GGML_ASSERT(a->ne[1] == ictx->nfft / 2 + 1);

    a = ggml_cont(ctx, ggml_permute(ctx, a, 1, 0, 2, 3));
    b = ggml_cont(ctx, ggml_permute(ctx, b, 1, 0, 2, 3));

    auto frames = ggml_add(ctx,
        ggml_mul_mat(ctx, ictx->idft_real, a),
        ggml_mul_mat(ctx, ictx->idft_imag, b));
    frames = ggml_mul(ctx, frames, c);

    auto t_idx = ggml_arange(ctx, 0, static_cast<float>(b->ne[1] * hop_len), static_cast<float>(hop_len));
    auto n_idx = ggml_arange(ctx, 0, static_cast<float>(ictx->nfft), 1.f);
    t_idx = ggml_reshape_2d(ctx, t_idx, 1, t_idx->ne[0]);
    t_idx = ggml_repeat_4d(ctx, t_idx, ictx->nfft, t_idx->ne[1], 1, 1);
    n_idx = ggml_repeat_4d(ctx, n_idx, n_idx->ne[0], a->ne[1], 1, 1);
    auto idx = ggml_add(ctx, t_idx, n_idx);
    idx = ggml_cast(ctx, idx, GGML_TYPE_I32);

    auto w_sq = ggml_mul(ctx, c, c);
    auto out_len_full = (b->ne[1] - 1) * hop_len + (center ? 0 : ictx->nfft);
    ggml_tensor* args[] = { idx, frames, w_sq, ggml_new_tensor_1d(ctx, GGML_TYPE_F32, out_len_full) };
    return ggml_custom_4d(ctx, GGML_TYPE_F32, out_len_full, 1, 1, 1, args, 4, ggml_istft_finalize_op, GGML_N_TASKS_MAX, reinterpret_cast<void*>(intptr_t(center ? ictx->nfft / 2 : 0)));
}
#endif
