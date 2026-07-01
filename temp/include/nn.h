#ifndef __NN_H__
#define __NN_H__


#include "ggml.h"

typedef enum nn_inference_buffer_policy
{
    NN_INFERENCE_BUFFER_POLICY_SHARED,    ///< Share buffers between the LLM KV cache and token2wav intermediates to minimize memory usage.
    NN_INFERENCE_BUFFER_POLICY_BALANCED,  ///< Share buffers, but keep reusable sequence segments in memory for faster restoration on the next run.
    NN_INFERENCE_BUFFER_POLICY_DEDICATED, ///< Use separate buffers for the LLM KV cache and token2wav intermediates to prioritize speed.
    NN_INFERENCE_BUFFER_POLICY_COUNT      // Sentinel value.
} nn_inference_buffer_policy_e;


void nn_log_callback_default(ggml_log_level level,  const char *text, void *user_data);

struct ggml_backend_op_capabilities
{
    bool concat_i32 : 1;
    bool repeat_f16 : 1;
};



#endif //  __NN_H__
