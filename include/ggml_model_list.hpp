#ifndef __GGML_MODEL_LIST_H__
#define __GGML_MODEL_LIST_H__

/* vad_silero */
int load_silero_model(struct ggml_handle_t *ggml_handle, const char *model_path);
int silero_inference(struct ggml_handle_t *ggml_handle, matrix_t **input_matrix,
                        matrix_t **output_matrix);

void unload_silero_model(struct ggml_handle_t *ggml_handle);


/* vad_fsmn */
int load_fsmn_model(struct ggml_handle_t *ggml_handle, const char *model_path);
int fsmn_inference(struct ggml_handle_t *ggml_handle, matrix_t **input_matrix,
                        matrix_t **output_matrix);

void unload_fsmn_model(struct ggml_handle_t *ggml_handle);


/* asr_sensevoice */

int load_sensevoice_model(struct ggml_handle_t *ggml_handle, const char *model_path);
int sensevoice_inference(struct ggml_handle_t *ggml_handle, matrix_t **input_matrix,
                        matrix_t **output_matrix);

void unload_sensevoice_model(struct ggml_handle_t *ggml_handle);


struct ggml_model_t
{
    char name[128];

    int (*load_model)(struct ggml_handle_t *ggml_handle, const char *model_path);

    int (*inference)(struct ggml_handle_t *ggml_handle, matrix_t **input_matrix, matrix_t **output_matrix);
    void (*unload_model)(struct ggml_handle_t *ggml_handle);
};


static struct ggml_model_t ggml_models[] =
{
    //name                  load                    inference                   unload
    {"SenseVoiceSmall",     load_silero_model,       silero_inference ,         unload_silero_model},
    //{"SenseVoiceSmall",     load_sensevoice_model,  sensevoice_inference ,      unload_sensevoice_model},
    {"FsmnVADStreaming",    load_fsmn_model,         fsmn_inference,            unload_fsmn_model},
};


#endif //  __GGML_MODEL_LIST_H__
