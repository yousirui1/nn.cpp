#ifndef __GGML_TTS_H__
#define __GGML_TTS_H__


#ifdef __cplusplus
extern "C"
{
#endif // __cplusplus



EXPORT void *tts_model_alloc(const char *model_path, void *user_data);
EXPORT int tts_model_inference(void *handle, char *input_data, int input_size, float **output_data, int *output_size, int n_thread);
EXPORT void tts_model_free(void *handle);


#ifdef __cplusplus
}
#endif // __cplusplus

#endif //  __GGML_TTS_H__
