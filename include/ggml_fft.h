#ifndef __GGML_FFT_H__
#define __GGML_FFT_H__


#include <ggml.h>

#include <functional>

#include <memory>

struct fft_context;
void delete_fft_context(fft_context* ctx);
struct fft_context_deleter { void operator()(fft_context* ctx) const { delete_fft_context(ctx); } };
using fft_context_ptr = std::unique_ptr<fft_context, fft_context_deleter>;
fft_context_ptr create_fft_context(int nfft);

void fft(const float* signal, float* buffer, fft_context& ctx);


ggml_tensor* ggml_fft(ggml_context* ctx, ggml_tensor* a, fft_context* fctx);
ggml_tensor* ggml_stft(ggml_context* ctx, ggml_tensor* a, ggml_tensor* b, int hop_len, bool center, fft_context* fctx);

struct istft_context;
void delete_istft_context(istft_context* ctx);
struct istft_context_deleter { void operator()(istft_context* ctx) const { delete_istft_context(ctx); } };
using istft_context_ptr = std::unique_ptr<istft_context, istft_context_deleter>;
istft_context_ptr create_istft_context(int nfft, ggml_context* ctx, std::function<void(ggml_tensor*, void*, size_t)> set_data);

ggml_tensor* ggml_istft(ggml_context* ctx, ggml_tensor* a, ggml_tensor* b, ggml_tensor* c, int hop_len, bool center, istft_context* ictx);



#endif //  __GGML_FFT_H__
