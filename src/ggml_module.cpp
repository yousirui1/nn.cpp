#include "base.h"
#include "gguf_loader.h"
#include "ggml_module.h"

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

void Module::onload(const gguf_loader& loader, const std::string& prefix) {}

void BasicModule::onload(const gguf_loader& loader, const std::string& prefix)
{
    LOAD_TENSOR(weight);
    LOAD_OPTIONAL_TENSOR(bias);
}

void LayerNorm::onload(const gguf_loader& loader, const std::string& prefix)
{
    LOAD_OPTIONAL_TENSOR(weight);
    LOAD_OPTIONAL_TENSOR(bias);
}

ggml_tensor* LayerNorm::build_cgraph(ggml_context* ctx, ggml_tensor* x) const
{
    //cuda ? 
    x = ggml_norm(ctx, x, eps);

    if (weight)
    {   
        x = ggml_mul(ctx, weight, x); 
        if (bias)
            x = ggml_add(ctx, x, bias);
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

ggml_tensor *Linear::build_cgraph(ggml_context *ctx, ggml_tensor *x) const
{
    auto out = ggml_mul_mat(ctx, weight, x);
    if(bias)
        out = ggml_add(ctx, out, bias);

    return out;
}

//to do 
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

    LOG_DEBUG("SenseVoiceEncoderSmall output_size %d linear_units %d attention_heads %d num_blocks %d tp_blocks %d ", output_size, linear_units, attention_heads, num_blocks, tp_blocks);

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

void CTC::onload(const gguf_loader &loader, const std::string &prefix)
{
    LOAD_OPTIONAL_TENSOR(weight);
    LOAD_OPTIONAL_TENSOR(bias);
}

std::array<ggml_tensor *, 2> CTC::build_cgraph(ggml_context *ctx, ggml_tensor *encoder_out) const
{
    // Reshape encoder_out to merge batch and time dimensions
    ggml_tensor *x ;
    {
        ggml_reshape_2d(ctx, encoder_out, encoder_out->ne[0], encoder_out->ne[1] * encoder_out->ne[2]);
        x = ggml_mul_mat(ctx, weight, x);
        x = ggml_add(ctx, x, bias);
        // Reshape back to 3D
        x = ggml_reshape_3d(ctx, x, x->ne[0], encoder_out->ne[1], encoder_out->ne[2]);
    }

    ggml_tensor *probs = ggml_soft_max(ctx, x);
    probs = ggml_reshape_2d(ctx, probs, probs->ne[0], probs->ne[1] * probs->ne[2] * probs->ne[3]);
    ggml_tensor *argmax_logit = ggml_argmax(ctx, probs);
    argmax_logit = ggml_reshape_3d(ctx, argmax_logit, x->ne[1], x->ne[2], x->ne[3]);
    return {probs, argmax_logit};
}
