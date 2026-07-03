#ifndef __GGML_MODEL_H__
#define __GGML_MODEL_H__

#include "matrix.h"

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
