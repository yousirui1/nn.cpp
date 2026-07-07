#ifndef __VAD_H__
#define __VAD_H__

#include <vector>
#include <string>

#ifdef __cplusplus
extern "C"
{
#endif // __cplusplus


struct vad_result_t
{
    float start_time;
    float end_time;
    //float *data;
    //int size;
};

EXPORT void vad_register_callback(void* handle, void (*cb)(void *, int , void *), void *user_data);

EXPORT void *vad_model_alloc(const char *model_path, void *user_data);
EXPORT int vad_model_inference(void *handle, float *input_data, int input_size, float **output_data, int *output_size, int n_threads);
EXPORT void vad_model_free(void *handle);


#ifdef __cplusplus
}
#endif // __cplusplus


#endif //  __ASR_H__
