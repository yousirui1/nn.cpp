#ifndef __GGML_COSYVOICE_H__
#define __GGML_COSYVOICE_H__


int load_cosyvoice_model(struct ggml_handle_t *ggml_handle, const char *model_path,
                        shape_t *input_shape, int *in_nodes,shape_t *output_shape, int *out_nodes);

int cosyvoice_inference(struct ggml_handle_t *ggml_handle, matrix_t **input_matrix, matrix_t **output_matrix, int n_threads, void *param);


void unload_cosyvoice_model();


#endif //  __GGML_COSYVOICE_H__
