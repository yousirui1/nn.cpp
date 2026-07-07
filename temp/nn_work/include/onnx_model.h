#ifndef __ONNX_MODEL_H__
#define __ONNX_MODEL_H__

#include "matrix.h"
#include "onnxruntime_c_api.h"

#ifdef __cplusplus
extern "C"
{
#endif // __cplusplus


typedef enum{
    USE_CPU = 0,
    USE_CUDA,
    USE_OPENVINO,
    USE_TENSOR_RT,
}DEVICE_TYPE;

static const char device_name[][128] = {
    "CPU",
    "CUDA",
    "OPENVINO",
    "TENSOR_RT"
};


#define MAX_NUM_LAYER 64

#define ORT_ERROR(expr)                             \
  do {                                                       \
    OrtStatus* onnx_status = (expr);                         \
    if (onnx_status != NULL) {                               \
      const char* msg = ort->GetErrorMessage(onnx_status); \
	  LOG_ERROR("%s", msg);									  \
      ort->ReleaseStatus(onnx_status);                     \
    }                                                        \
 } while(0);


struct onnx_handle_t
{
    OrtSession *session;

    OrtSessionOptions *session_options;
    OrtMemoryInfo *memory_info;

    size_t in_nodes;
    size_t out_nodes;

    OrtValue *input_tensor[MAX_NUM_LAYER];

    OrtValue *output_tensor[MAX_NUM_LAYER];

    char *input_names[MAX_NUM_LAYER];
    char *output_names[MAX_NUM_LAYER];

    int is_dynamic;
};

int init_onnx();
void deinit_onnx();

int onnx_use_gpu(int enable);

void *onnx_model_alloc(const char *model_data, uint64_t model_size, 
                        shape_t *input_shape, int *in_nodes,
                        shape_t *output_shape, int *out_nodes, void *user_data);


int onnx_data_alloc(void *handle,
        int in_nodes, shape_t *input_shape, matrix_t **input_matrix, int in_alloc_flag,                   int out_nodes, shape_t *output_shape, matrix_t **output_matrix, int out_alloc_flag);

int onnx_data_free(void *handle, matrix_t **input_matrix, matrix_t **output_matrix);
void onnx_model_free(void *handle);

int onnx_inference(void *handle, void *input_data, void *output_data, int is_debug);


#ifdef __cplusplus
}
#endif // __cplusplus



#endif //  __ONNX_MODEL_H__
