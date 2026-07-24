#ifndef __GGML_COSYVOICE_H__
#define __GGML_COSYVOICE_H__


int load_cosyvoice_model(struct ggml_handle_t *ggml_handle, const char *model_path, int model_size);

int cosyvoice_inference(struct ggml_handle_t *ggml_handle, matrix_t **input_matrix, matrix_t **output_matrix, void *param);

void unload_cosyvoice_model(struct ggml_handle_t *ggml_handle);


#endif //  __GGML_COSYVOICE_H__
