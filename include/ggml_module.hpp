#ifndef __GGML_MODULE_H__
#define __GGML_MODULE_H__

#include <vector>
#include <functional>
#include <array>
#include <memory>
#include <vector>
#include <string>
#include <map>
#include "ggml.h"
#include "ggml-backend.h"

struct gguf_loader;


struct ggml_backend_op_capabilities_t
{
    bool concat_i32 : 1;
    bool repeat_f16 : 1;
};


template<bool up = true>
constexpr size_t get_aligned_size(size_t size, size_t alignment)
{
    if constexpr (up)
        return (size + alignment - 1) & ~(alignment - 1);
    else
        return size & ~(alignment - 1);
}

struct Module
{
    virtual void onload(const gguf_loader& loader, const std::string& prefix);

    std::map<std::string, ggml_tensor**> tensors;
    std::map<std::string, Module*> submodules;

    std::map<std::string, ggml_tensor**> get_all_tensors()
    {
        auto all_tensors = tensors;
        for (const auto& [name, submodule] : submodules)
        {
            auto sub_tensors = submodule->get_all_tensors();
            for (auto& kv : sub_tensors)
                all_tensors.insert(std::move(kv));
        }
        return all_tensors;
    }
};

struct BasicModule : Module
{
    void onload(const gguf_loader& loader, const std::string& prefix);

    ggml_tensor* weight;
    ggml_tensor* bias;
};

struct Conv1d : BasicModule
{
    ggml_tensor* build_cgraph(ggml_context* ctx, ggml_tensor* x, int s, int p, int d, int g, ggml_backend_op_capabilities_t capabilities) const;
};

struct Linear : BasicModule
{
    ggml_tensor* build_cgraph(ggml_context* ctx, ggml_tensor* x) const;
};

struct LayerNorm : BasicModule
{
    constexpr static float eps = 1e-6f;

    void onload(const gguf_loader& loader, const std::string& prefix);

    ggml_tensor* build_cgraph(ggml_context* ctx, ggml_tensor* x) const;
};


struct STFT : Module
{
    ggml_tensor *forward_basis_buffer;

    void onload(const gguf_loader& loader, const std::string& prefix);

    ggml_tensor* build_cgraph(ggml_context* ctx, ggml_tensor* x) const;
};


struct LSTM : Module
{
    struct ggml_tensor *lstm_weight_ih;
    struct ggml_tensor *lstm_bias_ih;
    struct ggml_tensor *lstm_weight_hh;
    struct ggml_tensor *lstm_bias_hh;

    void onload(const gguf_loader& loader, const std::string& prefix);

    ggml_tensor* build_cgraph(ggml_context* ctx, ggml_tensor* x) const;
};


struct FSMNBlock : Module
{
    Linear linear;
    Conv1d fsmn_block;
    Linear affine;

    void onload(const gguf_loader &loader, const std::string &prefix);
    ggml_tensor* build_cgraph(ggml_context* ctx, ggml_tensor* x) const;
};


#endif //  __GGML_MODULE_H__
