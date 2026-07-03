#include "base.h"
#include "ggml_module.hpp"
#include "gguf_loader.hpp"
#include "ggml_handle.hpp"
#include "ggml_model_list.hpp"

//to do
int get_ggml_model(struct ggml_handle_t *ggml_handle)
{
    int i;
    for(i = 0; i < ARRAY_SIZE(ggml_models); i++)
    {   
        if(STRPREFIX(ggml_models[i].name, ggml_handle->model_name))
        {
            LOG_DEBUG("find name %s", ggml_handle->model_name);
            ggml_handle->load_model = ggml_models[i].load_model;
            ggml_handle->inference = ggml_models[i].inference;
            ggml_handle->unload_model = ggml_models[i].unload_model;
            return SUCCESS;
        }
    }   
    return ERROR;
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


void Module::onload(const gguf_loader& loader, const std::string& prefix) {}

void BasicModule::onload(const gguf_loader& loader, const std::string& prefix)
{
    LOAD_TENSOR(weight);
    LOAD_OPTIONAL_TENSOR(bias);
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
    struct ggml_tensor *in_lstm_hidden_state = ggml_new_tensor_1d(ctx, x->type, x->ne[1]);
    struct ggml_tensor *in_lstm_context = ggml_new_tensor_1d(ctx, x->type, x->ne[1]);

    struct ggml_tensor *out_lstm_hidden_state = nullptr;
    struct ggml_tensor *out_lstm_context = nullptr;

    ggml_set_name(in_lstm_context, "in_lstm_context");
    ggml_set_name(in_lstm_hidden_state, "in_lstm_hidden_state");

    x = ggml_cont(ctx, ggml_transpose(ctx, x));

    struct ggml_tensor *gates = ggml_add(ctx,
            ggml_add(ctx, ggml_mul_mat(ctx, lstm_weight_ih, x), lstm_bias_ih),
            ggml_add(ctx, ggml_mul_mat(ctx, lstm_weight_hh, in_lstm_hidden_state), lstm_bias_hh));

    ggml_set_name(gates, "gates");

    struct ggml_tensor *input_gates = ggml_sigmoid(ctx, ggml_view_2d(ctx, gates, gates->ne[0] /4, gates->ne[1], gates->nb[1], 0));
    struct ggml_tensor *forget_gates = ggml_sigmoid(ctx, ggml_view_2d(ctx, gates, gates->ne[0] /4, gates->ne[1], gates->nb[1], gates->nb[0] / 4 * gates->ne[0]));
    struct ggml_tensor *cell_gate = ggml_tanh(ctx, ggml_view_2d(ctx, gates, gates->ne[0] /4, gates->ne[1], gates->nb[1], 2 * gates->nb[0] / 4 * gates->ne[0]));
    struct ggml_tensor *out_gates = ggml_sigmoid(ctx, ggml_view_2d(ctx, gates, gates->ne[0] /4, gates->ne[1], gates->nb[1], 3 * gates->nb[0] / 4 * gates->ne[0]));

    ggml_set_name(input_gates, "input_gates");
    ggml_set_name(forget_gates, "forget_gates");
    ggml_set_name(cell_gate, "cell_gate");
    ggml_set_name(out_gates, "out_gates");

    out_lstm_context = ggml_add(ctx,
                        ggml_mul(ctx, forget_gates, in_lstm_context),
                        ggml_mul(ctx, input_gates, cell_gate));

    ggml_set_name(out_lstm_context, "out_lstm_context");
    out_lstm_hidden_state = ggml_mul(ctx, out_gates, ggml_tanh(ctx, out_lstm_context));
    ggml_set_name(out_lstm_hidden_state, "out_lstm_hidden_state");
   
    return out_lstm_hidden_state;
}

ggml_tensor *Linear::build_cgraph(ggml_context *ctx, ggml_tensor *x) const 
{
    auto out = ggml_mul_mat(ctx, weight, x);
    if(bias)
        out = ggml_add(ctx, out, bias);

    return out;
}

void FSMNBlock::onload(const gguf_loader& loader, const std::string& prefix)
{
    LOAD_SUBMODULE_EX("linear.linear", linear);
    LOAD_SUBMODULE_EX("fsmn_block.conv_left", fsmn_block);
    LOAD_SUBMODULE_EX("affine.linear", affine);
}

ggml_tensor *FSMNBlock::build_cgraph(ggml_context *ctx, ggml_tensor *x) const
{

#if 0
    struct ggml_tensor *in_lstm_hidden_state = ggml_new_tensor_1d(ctx, x->type, x->ne[1]);
    struct ggml_tensor *in_lstm_context = ggml_new_tensor_1d(ctx, x->type, x->ne[1]);

    struct ggml_tensor *out_lstm_hidden_state = nullptr;
    struct ggml_tensor *out_lstm_context = nullptr;

    ggml_set_name(in_lstm_context, "in_lstm_context");
    ggml_set_name(in_lstm_hidden_state, "in_lstm_hidden_state");

    x = ggml_cont(ctx, ggml_transpose(ctx, x));

    struct ggml_tensor *gates = ggml_add(ctx,
            ggml_add(ctx, ggml_mul_mat(ctx, lstm_weight_ih, x), lstm_bias_ih),
            ggml_add(ctx, ggml_mul_mat(ctx, lstm_weight_hh, in_lstm_hidden_state), lstm_bias_hh));

    ggml_set_name(gates, "gates");

    struct ggml_tensor *input_gates = ggml_sigmoid(ctx, ggml_view_2d(ctx, gates, gates->ne[0] /4, gates->ne[1], gates->nb[1], 0));
    struct ggml_tensor *forget_gates = ggml_sigmoid(ctx, ggml_view_2d(ctx, gates, gates->ne[0] /4, gates->ne[1], gates->nb[1], gates->nb[0] / 4 * gates->ne[0]));
    struct ggml_tensor *cell_gate = ggml_tanh(ctx, ggml_view_2d(ctx, gates, gates->ne[0] /4, gates->ne[1], gates->nb[1], 2 * gates->nb[0] / 4 * gates->ne[0]));
    struct ggml_tensor *out_gates = ggml_sigmoid(ctx, ggml_view_2d(ctx, gates, gates->ne[0] /4, gates->ne[1], gates->nb[1], 3 * gates->nb[0] / 4 * gates->ne[0]));

    ggml_set_name(input_gates, "input_gates");
    ggml_set_name(forget_gates, "forget_gates");
    ggml_set_name(cell_gate, "cell_gate");
    ggml_set_name(out_gates, "out_gates");

    out_lstm_context = ggml_add(ctx,
                        ggml_mul(ctx, forget_gates, in_lstm_context),
                        ggml_mul(ctx, input_gates, cell_gate));

    ggml_set_name(out_lstm_context, "out_lstm_context");
    out_lstm_hidden_state = ggml_mul(ctx, out_gates, ggml_tanh(ctx, out_lstm_context));
    ggml_set_name(out_lstm_hidden_state, "out_lstm_hidden_state");
   
    return out_lstm_hidden_state;
#endif
    return x;
}


