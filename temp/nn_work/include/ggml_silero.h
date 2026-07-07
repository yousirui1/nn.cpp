#ifndef __GGML_SILERO_H__
#define __GGML_SILERO_H__


struct silero_params_t
{
    uint32_t n_encoder_layer;
    uint32_t lstm_state_dim;

    uint32_t n_batch;

    float threshold;
    float neg_threshold;
    int32_t min_speech_duration_ms;
    int32_t max_speech_duration_ms;
    int32_t min_silence_duration_ms;
    int32_t speech_pad_ms;
};

int load_silero_model(struct ggml_handle_t *ggml_handle, const char *model_path,
                        shape_t *input_shape, int *in_nodes,shape_t *output_shape, int *out_nodes);

int silero_frontend_process(struct ggml_handle_t *ggml_handle);
int silero_backend_process(struct ggml_handle_t *ggml_handle, float speech_prob);

int silero_inference(struct ggml_handle_t *ggml_handle,
                                matrix_t **input_matrix,
                                matrix_t **output_matrix,
                                const int n_threads, void *param);


#endif //  __GGML_SILERO_H__
