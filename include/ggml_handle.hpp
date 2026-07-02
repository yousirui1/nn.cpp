#ifndef __GGML_HANDLE_H__
#define __GGML_HANDLE_H__

#include "matrix.h"
#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"

#define MAX_NUM_LAYER 64 // to do
                         //
#ifdef __cplusplus
extern "C" 
{
#endif // __cplusplus

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

void set_graph_backend(struct ggml_cgraph* gf, ggml_backend_sched_t sched, ggml_backend_t backend, ggml_backend_t cpu_backend, int nodes);

bool ggml_graph_compute_helper(ggml_backend_sched_t sched, struct ggml_cgraph *graph, int n_threads);
int get_ggml_model(struct ggml_handle_t *ggml_handle);

#ifdef __cplusplus
}
#endif // __cplusplus

#endif //  __GGML_HANDLE_H__
