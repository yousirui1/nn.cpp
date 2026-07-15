#ifndef __GGML_MODEL_H__
#define __GGML_MODEL_H__

#include <map>
#include <string>
#include "matrix.h"
#include "ggml-backend.h"


#define MAX_NUM_LAYER 64

struct ggml_handle_t
{
    char model_name[128];

    int in_nodes;
    int out_nodes;

    int n_thread;
    int use_gpu;

    int is_dynamic;
    int is_dup;

    ggml_backend_t backend;
    ggml_backend_t cpu_backend;

    char *input_names[MAX_NUM_LAYER];
    char *output_names[MAX_NUM_LAYER];

    struct shape_t input_shape[MAX_NUM_LAYER];
    struct shape_t output_shape[MAX_NUM_LAYER];

    int (*load_model)(struct ggml_handle_t *ggml_handle, const char *model_path, int model_size);

    int (*inference)(struct ggml_handle_t *ggml_handle, matrix_t **input_matrix, matrix_t **output_matrix, void *param);
    void (*unload_model)(struct ggml_handle_t *ggml_handle);

    void (*cb)(void *data, int size, void *user_data); 


    void *params;
    void *model;
    void *state;
    void *user_data;
};

int ggml_model_weight_alloc(ggml_context *ctx, ggml_backend_t dev_backend, ggml_backend_buffer_t &buffer, const std::map<std::string, ggml_tensor**> &tensors);

void set_graph_backend(struct ggml_cgraph* gf, ggml_backend_sched_t sched, ggml_backend_t backend, ggml_backend_t cpu_backend, int nodes); 

bool ggml_graph_compute_helper(ggml_backend_sched_t sched, struct ggml_cgraph *graph, int n_threads);

int init_ggml();
void deinit_ggml();


struct ggml_handle_t *ggml_model_alloc(const char *model_data, uint64_t model_size,
                        shape_t *input_shape, int *in_nodes,
                        shape_t *output_shape, int *out_nodes, int n_thread, void *user_data);

void ggml_model_free(struct ggml_handle_t *ggml_handle);

int ggml_data_alloc(struct ggml_handle_t *ggml_handle,
         int in_nodes, shape_t *input_shape, matrix_t **input_matrix, int in_alloc_flag,
         int out_nodes, shape_t *output_shape, matrix_t **output_matrix, int out_alloc_flag);

int ggml_data_free(struct ggml_handle_t *ggml_handle, matrix_t **input_matrix, matrix_t **output_matrix);


void ggml_print_tensor(struct ggml_tensor * tensor);

int ggml_inference(struct ggml_handle_t *ggml_handle, matrix_t **input_matrix, matrix_t **output_matrix, void *param, int is_debug);



#endif //  __GGML_MODEL_H__
