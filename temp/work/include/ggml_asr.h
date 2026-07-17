#ifndef __GGML_ASR_H__
#define __GGML_ASR_H__

#ifdef __cplusplus
extern "C"
{
#endif // __cplusplus


EXPORT void *asr_model_alloc(const char *model_path, void *user_data);
EXPORT int asr_model_inference(void *handle, float *input_data, int input_size, char **output_data, int *output_size, int n_thread);
EXPORT void asr_model_free(void *handle);


#ifdef __cplusplus
}
#endif // __cplusplus


#endif //  __GGML_ASR_H__
