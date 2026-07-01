#ifndef __GGML_MODEL_H__
#define __GGML_MODEL_H__

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"



//to do 
//strcut ggml_backend_op_capabilities_t
//{
//
//};

struct ggml_handle_t
{
    char model_name[128];

    int in_nodes;
    int out_nodes;

    int n_thread;
    int use_gpu;

    int is_dynamic;
    
    ggml_backend_t backend; 
    ggml_backend_t cpu_backend;

    char *input_names[MAX_NUM_LAYER];
    char *output_names[MAX_NUM_LAYER];

    struct shape_t input_shape[MAX_NUM_LAYER];
    struct shape_t output_shape[MAX_NUM_LAYER];
    //to do 
    //matrix_t *input_matrix[MAX_NUM_LAYER];
    //matrix_t *output_matrix[MAX_NUM_LAYER];
    
    int (*load_model)(struct ggml_handle_t *ggml_handle, const char *model_path);
    
    int (*inference)(struct ggml_handle_t *ggml_handle, matrix_t **input_matrix, matrix_t **output_matrix);
    void (*unload_model)(struct ggml_handle_t *ggml_handle);

    void (*cb)(void *data, int size, void *user_data);

    void *params;
    void *model;
    void *state;
    void *user_data;

    int device_type;
};



inline static const char *get_tensor_type_string(int type)
{
    switch (type)
    {
        case GGML_TYPE_F16:
            return "FP16";
        case GGML_TYPE_F32:
            return "FP32";
        case GGML_TYPE_I8:
            return "INT8";
        case GGML_TYPE_I16:
            return "INT16";
        case GGML_TYPE_I32:
            return "INT32";
        case GGML_TYPE_I64:
            return "INT64";
        default:
            return "Q_K";
    }
}

//to do 
int init_ggml();
void deinit_ggml();

int ggml_use_gpu(void *handle, int enable);

void *ggml_model_alloc(const char *model_data, uint64_t model_size,
                        shape_t *input_shape, int *in_nodes,
                        shape_t *output_shape, int *out_nodes, int n_thread, void *user_data);

int ggml_data_alloc(void *handle,
        int in_nodes, shape_t *input_shape, matrix_t **input_matrix, int in_alloc_flag,                   int out_nodes, shape_t *output_shape, matrix_t **output_matrix, int out_alloc_flag);

int ggml_data_free(void *handle, matrix_t **input_matrix, matrix_t **output_matrix);
void ggml_model_free(void *handle);

int ggml_inference(void *handle, matrix_t **input_matrix, matrix_t **output_matrix, int is_debug);


#endif //  __GGML_MODEL_H__
