#ifndef __GGML_MODULE_H__
#define __GGML_MODULE_H__

#include <vector>
#include <functional>
#include <array>
#include <memory>
#include <vector>
#include <string>
#include <map>
#include <ggml.h>
#include <ggml-alloc.h>
#include <ggml-backend.h>

struct gguf_loader;

struct ggml_backend_op_capabilities_t
{
    bool concat_i32    : 1;
    bool repeat_f16    : 1;
    bool pad           : 1;
    bool pad_reflect_1d: 1;
    bool im2col_f16    : 1;
    bool fill          : 1;
    bool cumsum        : 1;
    bool emb_cast_f32  : 1;
    bool top_k         : 1;
    bool leaky_relu    : 1;
    bool sin           : 1;
    bool cos           : 1;
    bool arange        : 1;
    bool elu           : 1;
    bool abs           : 1;
    bool floor         : 1;
    bool acc           : 1;

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

struct Linear : BasicModule
{
    ggml_tensor* build_cgraph(ggml_context* ctx, ggml_tensor* x) const;
};

struct LayerNorm : BasicModule
{
    constexpr static float eps = 1e-6f;
    void onload(const gguf_loader &loader, const std::string &prefix);
    ggml_tensor* build_cgraph(ggml_context* ctx, ggml_tensor* x) const;
};

struct Conv1d : BasicModule
{
    ggml_tensor* build_cgraph(ggml_context* ctx, ggml_tensor* x, int s, int p, int d, int g, ggml_backend_op_capabilities_t capabilities) const;
};


struct CausalConv1dBase : Conv1d
{
    virtual ~CausalConv1dBase() = default;
    virtual ggml_tensor *build_cgraph(ggml_context *ctx, ggml_tensor *x) const = 0;
};

struct CausalConv1d : CausalConv1dBase
{
    enum causal_type_t : char
    {
        left,
        right
    } causal_type;
    int d = 1;

    int causal_padding() const;

    ggml_tensor *build_cgraph(ggml_context *ctx, ggml_tensor *x) const;
};

struct CausalConvRNNF0Predictor : Module
{
    CausalConvRNNF0Predictor();

    CausalConv1d condnet_0;
    CausalConv1d condnet_2;
    CausalConv1d condnet_4;
    CausalConv1d condnet_6;
    CausalConv1d condnet_8;

    Linear classifier;

    void onload(const gguf_loader &loader, const std::string &prefix);
    ggml_tensor *build_cgraph(ggml_context *ctx, ggml_tensor *x) const;
};

struct CausalConv1dDownSample : CausalConv1dBase
{
    int s;
    ggml_tensor *build_cgraph(ggml_context *ctx, ggml_tensor *x) const;
};

struct CausalConv1dUpSample : CausalConv1dBase
{
    int s = 1;
    ggml_tensor *build_cgraph(ggml_context *ctx, ggml_tensor *x) const;
};

struct CausalConvPositionEmbedding : Module
{
    Conv1d conv1;
    Conv1d conv2;

    void onload(const gguf_loader &loader, const std::string &prefix);
    ggml_tensor *build_cgraph(ggml_context *ctx, ggml_tensor *x, ggml_backend_op_capabilities_t capabilities) const;

};

struct Snake : Module
{
    static constexpr float no_div_by_zero = 0.000000001f;
    ggml_tensor *alpha;

    void onload(const gguf_loader &loader, const std::string &prefix);
    ggml_tensor *build_cgraph(ggml_context *ctx, ggml_tensor *x) const;
};

struct ResBlock : Module
{
    std::vector<std::tuple<Snake, CausalConv1d, Snake, CausalConv1d>> convs;

    void onload(const gguf_loader &loader, const std::string &prefix);
    ggml_tensor *build_cgraph(ggml_context *ctx, ggml_tensor *x) const;
};


struct STFT : Module
{
    ggml_tensor *forward_basis_buffer;

    void onload(const gguf_loader& loader, const std::string& prefix);
    ggml_tensor* build_cgraph(ggml_context* ctx, ggml_tensor* x) const;
};

struct LSTM : Module
{
    ggml_tensor *lstm_weight_ih;
    ggml_tensor *lstm_bias_ih;
    ggml_tensor *lstm_weight_hh;
    ggml_tensor *lstm_bias_hh;

    void onload(const gguf_loader& loader, const std::string& prefix);

    ggml_tensor* build_cgraph(ggml_context* ctx, ggml_tensor* x) const;
};



struct SinusoidalPositionEncoder : Module
{
    ggml_tensor* weight;

    void compute(int , int , ggml_tensor *position, ggml_tensor *embedding);

    void onload(const gguf_loader &loader, const std::string &prefix);
    ggml_tensor *build_cgraph(ggml_context *ctx, ggml_tensor *feature, int n_hidden_state) const;
};

struct SinusPositionEmbedding : Module
{
    ggml_tensor *emb = nullptr;
    
    ggml_tensor *build_cgraph(ggml_context *ctx, ggml_tensor *x) const;
};

struct PositionwiseFeedForward : Module
{
    Linear w_1;
    Linear w_2;

    void onload(const gguf_loader &loader, const std::string &prefix);
    ggml_tensor *build_cgraph(ggml_context *ctx, ggml_tensor *x) const;
};


struct InputEmbedding : Module
{
    Linear proj;

    CausalConvPositionEmbedding conv_pos_embed;

    void onload(const gguf_loader &loader, const std::string &prefix);

    ggml_tensor *build_cgraph(ggml_context *ctx, ggml_tensor *x, ggml_tensor *cond, ggml_tensor *text_embed, ggml_tensor *spks, ggml_backend_op_capabilities_t capabilities) const;
};

struct TimestepEmbedding : Module
{
    SinusPositionEmbedding time_embed;
    Linear time_mlp_0;
    Linear time_mlp_2;

    void onload(const gguf_loader &loader, const std::string &prefix);
    ggml_tensor *build_cgraph(ggml_context *ctx, ggml_tensor *x) const;
};

struct CTC : BasicModule
{
    void onload(const gguf_loader &loader, const std::string &prefix);
    std::array<ggml_tensor *, 2>build_cgraph(ggml_context *ctx, ggml_tensor *x) const;
};


struct AdaLayerNormFinal : Module
{
    Linear linear;

    LayerNorm norm;

    void onload(const gguf_loader &loader, const std::string &prefix);

    ggml_tensor *build_cgraph(ggml_context *ctx, ggml_tensor *x, ggml_tensor *emb) const;
};

struct AdaLayerNormZero : Module
{
    Linear linear;

    LayerNorm norm;

    void onload(const gguf_loader &loader, const std::string &prefix);
    std::array<ggml_tensor *, 5> build_cgraph(ggml_context *ctx, ggml_tensor *x, ggml_tensor *emb) const;
};

struct Attention : Module
{
    int heads;
    bool fattn;

    Linear to_q;
    Linear to_k;
    Linear to_v;
    Linear to_out;

    void onload(const gguf_loader &loader, const std::string &prefix);

    ggml_tensor *build_cgraph(ggml_context *ctx, ggml_tensor *x, ggml_tensor *position_ids, int64_t cut_len) const;
};

struct MultiHeadedAttentionSANM : Module
{
    Linear linear_out;
    Linear linear_q;
    Linear linear_k;
    Linear linear_v;

    Conv1d fsmn_block;

    void onload(const gguf_loader &loader, const std::string &prefix);
    ggml_tensor *build_cgraph(ggml_context *ctx, ggml_tensor *x, int n_hidden_state, int n_head, int fsmn_kerenl_size, int flash_attn) const;
};

struct FSMNBlock : Module
{
    Linear linear;
    Conv1d fsmn_block;
    Linear affine;

    void onload(const gguf_loader &loader, const std::string &prefix);
    ggml_tensor* build_cgraph(ggml_context* ctx, ggml_tensor* x) const;
};


struct FeedForward : Module
{
    Linear ff_0_0;
    Linear ff_2;

    void onload(const gguf_loader &loader, const std::string &prefix);
    ggml_tensor *build_cgraph(ggml_context *ctx, ggml_tensor *x) const;
};

struct DiTBlock : Module
{
    AdaLayerNormZero attn_norm;
    Attention attn;

    LayerNorm ff_norm;
    FeedForward ff;

    void onload(const gguf_loader &loader, const std::string &prefix);
    ggml_tensor *build_cgraph(ggml_context *ctx, ggml_tensor *x, ggml_tensor *time_emb, ggml_tensor *position_ids, int64_t cut_len) const;
};

struct DiT : Module
{
    TimestepEmbedding time_embed;
    InputEmbedding input_embed;

    std::vector<DiTBlock> transformer_blocks;

    AdaLayerNormFinal norm_out;
    Linear proj_out;

    int mel_dim;

    void onload(const gguf_loader &loader, const std::string &prefix);
    ggml_tensor *build_cgraph(ggml_context *ctx, ggml_tensor *x, ggml_tensor *mu, ggml_tensor *t,
            ggml_tensor *spks, ggml_tensor *cond, int64_t cut_len, ggml_tensor* &position_ids,
            ggml_backend_op_capabilities_t capabilities) const;

};

struct CausalConditionalCFM : Module
{
    std::array<float, 11> t_span;
    float inference_cfg_rate;

    DiT estimator;

    void onload(const gguf_loader &loader, const std::string &prefix);

    struct DiTContext
    {
        ggml_tensor *x;
        ggml_tensor *mu_in;
        ggml_tensor *spks_in;
        ggml_tensor *cond_in;
    };

    DiTContext prepare_context(ggml_context *ctx, ggml_tensor *mu, ggml_tensor *spks, ggml_tensor *cond) const;
    std::array<float, 2> get_t_and_dt(ggml_context *ctx, int step) const;

    ggml_tensor* build_cgraph_one_step(ggml_context* ctx, const DiTContext& dit_ctx, int step, ggml_backend_op_capabilities_t capabilities, int64_t cut_len, ggml_tensor*& t_tensor, ggml_tensor*& position_ids) const;


};

struct PreLookaheadLayer : Module
{
    int pre_lookahead_len;

    Conv1d conv1;
    Conv1d conv2;

    void onload(const gguf_loader &loader, const std::string &prefix);

    ggml_tensor *build_cgraph(ggml_context *ctx, ggml_tensor *inputs) const;
};

struct SineGen2 : Module
{
    ggml_tensor *rand_ini;

    std::array<ggml_tensor *, 2>build_cgraph(ggml_context *ctx, ggml_tensor *f0, 
                           int harmonic_num, int sampling_rate, int upsample_scale, float sine_amp,
                           int voiced_threshold, float noise_std) const;
};

struct SourceModuleHnNSF : Module
{
    SineGen2 l_sin_gen;
    Linear l_linear;

    void onload(const gguf_loader &loader, const std::string &prefix);

    std::array<ggml_tensor*, 2> build_cgraph(ggml_context* ctx, ggml_tensor* x, int harmonic_num, 
                            int sampling_rate, int upsample_scale, float sine_amp, 
                            int voiced_threshold, float noise_std) const;

};

struct EncoderLayerSANM : Module
{
    MultiHeadedAttentionSANM self_attn;

    PositionwiseFeedForward feed_forward;

    LayerNorm norm1;
    LayerNorm norm2;

    void onload(const gguf_loader &loader, const std::string &prefix);
    ggml_tensor *build_cgraph(ggml_context *ctx, ggml_tensor *x, int n_hidden_state, int n_head, int fsmn_kerenl_size, int flash_attn) const;
};


struct Qwen2MLP : Module
{
    Linear gate_proj;
    Linear up_proj;
    Linear down_proj;
    
    void onload(const gguf_loader &loader, const std::string &prefix);
    ggml_tensor *build_cgraph(ggml_context *ctx, ggml_tensor *x) const;
};

struct Qwen2RMSNorm: Module
{
    ggml_tensor *weight;

    void onload(const gguf_loader &loader, const std::string &prefix);
    ggml_tensor *build_cgraph(ggml_context *ctx, ggml_tensor *hidden_states, float variance_epsilon) const; 
};

struct Qwen2Attention : Module
{
    Linear q_proj;
    Linear k_proj;
    Linear v_proj;
    Linear o_proj;

    void onload(const gguf_loader &loader, const std::string &prefix);
};

struct Qwen2DecoderLayer : Module
{
    Qwen2Attention self_attn;
    Qwen2MLP mlp;
    Qwen2RMSNorm input_layernorm;
    Qwen2RMSNorm post_attention_layernorm;

    void onload(const gguf_loader &loader, const std::string &prefix);
};

struct SenseVoiceEncoderSmall : Module
{
    int output_size;
    int linear_units;
    int attention_heads;
    int num_blocks;
    int tp_blocks;

    SinusoidalPositionEncoder embed;

    EncoderLayerSANM encoders0;

    std::vector<EncoderLayerSANM> encoders;
    std::vector<EncoderLayerSANM> tp_encoders;

    LayerNorm after_norm;
    LayerNorm tp_norm;

    void onload(const gguf_loader &loader, std::string prefix);
    ggml_tensor *build_cgraph(ggml_context *ctx, ggml_tensor *x, int fsmn_kernel_size, int flash_attn) const;
};

#if 0
struct CausalMaskedDiffWithDiT : Module
{
    struct EncodeResult
    {
        ggml_tensor *mu;
        ggml_tensor *spk;
        ggml_tensor *conds;
        int64_t cut_len;
    };

    int token_mel_ratio;
    ggml_tensor *input_embeddig;
    Linear spk_embed_affine_layer;
    PreLookaheadLayer pre_lookahead_layer;
    CausalConditionalCFM decoder;
    
    void onload(const gguf_loader &loader, const std::string &prefix);

    //EncodeResult build_cgraph_encode(ggml_context *ctx, ggml_tensor *token, ggml_tensor *prompt_token, ggml_tensor *prompt_feat, ggml_tensor *embedding, ggml_back);
};
#endif

struct CausalHiFTGenerator : Module
{
    float lrelu_slope;
    int scale_factor;
    int nb_harmonics;
    int samping_rate;
    float nfs_alpha;
    int nsf_voiced_threshold;
    float nsf_sigma;
    float audio_limit;
    int nfft;
    int hop_len;

    //void fft_handle
    //void ifft_handle 

    ggml_tensor *window;
    CausalConvRNNF0Predictor f0_predictor;
    SourceModuleHnNSF m_source;
    CausalConv1d conv_pre;
    std::vector<CausalConv1dUpSample> ups;
    std::vector<std::unique_ptr<CausalConv1dBase>> source_downs;
    std::vector<ResBlock> source_resblocks;
    std::vector<ResBlock> resblocks;
    CausalConv1d conv_post;

    void onload(const gguf_loader &loader, const std::string &prefix);
    std::array<ggml_tensor *, 2> build_cgraph(ggml_context *ctx, ggml_tensor *speech_feat) const;
};

struct CosyVoice3LM : Module
{
    int num_attention_heads;
    int num_key_value_heads;
    float rms_norm_eps;
    float rope_theta;

    int sos_token_id;
    int task_token_id;

    ggml_tensor *embed_token_weight;
    ggml_tensor *speech_embedding_weight;
    std::vector<Qwen2DecoderLayer> layers;
    Qwen2RMSNorm norm;
    Linear llm_decoder;

    //void onload(const gguf_loader &loader, const std::string &prefix, const cosyvoice3_llm_params_t &params);

};

#endif //  __GGML_MODULE_H__

