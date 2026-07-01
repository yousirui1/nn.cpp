#ifndef __MODEL_H__
#define __MODEL_H__

#include "base.h"
#include "matrix.h"
#include "queue.h"

#define MAX_NUM_LAYER 64
#define MAX_MODEL 4

#define ONNX_MODEL_DIR "/home/ysr/project/models/onnx/"

struct model_handle_t
{
    int (*init)();
    void (*deinit)();

    int (*use_gpu)(int enable);

    void* (*alloc)(const char *model_data, uint64_t model_size, struct shape_t *input_shape, int *in_nodes,  struct shape_t *output_shape, int *out_nodes, int n_thread, void *user_data);

    //void *(*dup_context)(); to do 

    int (*data_alloc)(void *handle,
            int in_nodes, shape_t *input_shape, matrix_t **input_matrix, int in_alloc_flag,
            int out_nodes, shape_t *output_shape, matrix_t **output_matrix, int out_alloc_flag);

    int (*data_free)(void *handle, matrix_t **input_matrix, matrix_t **output_matrix);

    void (*free)(void *handle);

    int (*inference)(void *handle, matrix_t **input_matrix, matrix_t **output_matrix, int is_debug);

    void *handle;
    void *user_data;

    int in_nodes;
    int out_nodes;
    
    shape_t input_shape[MAX_NUM_LAYER];
    shape_t output_shape[MAX_NUM_LAYER];

    matrix_t *input_matrix[MAX_NUM_LAYER];
    matrix_t *output_matrix[MAX_NUM_LAYER];

    char *labels[128];
    int num_classes;
};

struct model_session_t
{
    struct model_handle_t models[MAX_MODEL];
    int n_model;

    int timeout;

    void *feature;
    void *tokenizer;

    uint8_t *queue_buf;
    struct queue_t queue;

    int is_running;
    pthread_t tid;

    float threshold;

    void (*cb)(void *data, int size, void *user_data);
    //void (*loop)(void *handle, int timeout);
    int (*inference)(struct model_session_t *model_session, void *input_data, int input_size, void *output_data, int output_size, void *param);

    void *user_data;
    void *handle; //to do 
};


typedef void (*inference_fn) (void *, int, void *, int, void *);

void get_model(struct model_handle_t *model);

int start_model_thread(struct model_session_t *model_session, int timeout, int cache_size);
void stop_model_thread(struct model_session_t *model_session);
void model_register_callback(struct model_session_t *model_session, void *cb, void *user_data);

int classes_top_k(float *output_data, int output_size, int top_k);
float mse_loss(float *input_data, float *output_data, int data_size);

typedef enum{
    USE_CPU = 0,
    USE_CUDA,
    USE_OPENVINO,
    USE_TENSOR_RT,
    USE_RKNN,
}DEVICE_TYPE;

static const char device_name[][128] = {
    "CPU",
    "CUDA",
    "OPENVINO",
    "TENSOR_RT",
    "RKNN",
};


#endif //  __MODEL_H__
