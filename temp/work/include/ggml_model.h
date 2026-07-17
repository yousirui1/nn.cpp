#ifndef __GGML_MODEL_H__
#define __GGML_MODEL_H__

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "matrix.h"

#define MAX_NUM_LAYER 64

struct ggml_backend_op_capabilities
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


struct ggml_handle_t
{
    char model_name[128];
    int in_nodes;
    int out_nodes;

    int n_thread; //to do 

    int use_gpu; //to do;

    char *input_names[MAX_NUM_LAYER];
    char *output_names[MAX_NUM_LAYER];

    int is_dynamic;

    ggml_backend_t backend; 
    ggml_backend_t cpu_backend; 
    
    shape_t input_shape[MAX_NUM_LAYER];
    shape_t output_shape[MAX_NUM_LAYER];

    matrix_t *input_matrix[MAX_NUM_LAYER];
    matrix_t *output_matrix[MAX_NUM_LAYER];

    struct ggml_backend_op_capabilities op_caps;

    //to do 
    int (*load_model)(struct ggml_handle_t *ggml_handle, const char *model_path,
                        shape_t *input_shape, int *in_nodes,shape_t *output_shape, int *out_nodes);

    int (*inference)(struct ggml_handle_t *ggml_handle, matrix_t **input_matrix, matrix_t **output_matrix, int n_threads, void *param);
    void (*unload_model)(struct ggml_handle_t *ggml_handle);

    void (*cb)(void *data, int size, void *user_data);

    void *params;
    void *model;
    void *state;
    void *user_data;
};


struct ggml_model_t
{
    char name[128];

    int (*load_model)(struct ggml_handle_t *ggml_handle, const char *model_path,
                        shape_t *input_shape, int *in_nodes,shape_t *output_shape, int *out_nodes);

    int (*inference)(struct ggml_handle_t *ggml_handle, matrix_t **input_matrix, matrix_t **output_matrix, int n_threads, void *param);
    void (*unload_model)(struct ggml_handle_t *ggml_handle);
};


bool ggml_graph_compute_helper(ggml_backend_sched_t sched, struct ggml_cgraph * graph, int n_threads);

void set_graph_backend(ggml_cgraph* gf, ggml_backend_sched_t sched, ggml_backend_t backend, ggml_backend_t cpu_backend, int nodes);

struct ggml_handle_t* ggml_model_alloc(const char *model_data, uint64_t model_size, void *user_data);

int ggml_data_alloc(struct ggml_handle_t *ggml_handle,
         int in_nodes, shape_t *input_shape, matrix_t **input_matrix, int in_alloc_flag,
         int out_nodes, shape_t *output_shape, matrix_t **output_matrix, int out_alloc_flag);

void ggml_model_free(struct ggml_handle_t *ggml_handle);

void ggml_print_tensor(struct ggml_tensor * tensor);

#endif //  __GGML_MODEL_H__
