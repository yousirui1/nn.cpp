#include "base.h"
#include "ggml_vad.h"
#include "ggml_model.h"
#include "ggml_silero.h"

static struct ggml_model_t ggml_models[] =
{
    //name        load                inference           unload
    {"SenseVoiceSmall", load_silero_model, silero_inference, NULL},
    //{"fsmnt",  load_silero_model, silero_build_graph, silero_inference},
};

static int get_ggml_model(struct ggml_handle_t *ggml_handle)
{
    int i;

    for(i = 0; i < ARRAY_SIZE(ggml_models); i++)
    {
        if(STRPREFIX(ggml_models[i].name, ggml_handle->model_name))
        {
            LOG_DEBUG("find vad  model name %s", ggml_handle->model_name);
            ggml_handle->load_model = ggml_models[i].load_model;
            ggml_handle->inference = ggml_models[i].inference;
            //ggml_handle->unload_model = ggml_modules[i].unload_model;
            return SUCCESS;
        }
    }
    return ERROR;
}


void vad_register_callback(void* handle, void (*cb)(void *, int , void *), void *user_data)
{
    struct ggml_handle_t *ggml_handle = (struct ggml_handle_t *)handle;
    if(ggml_handle)
    {
        ggml_handle->cb = cb;
        ggml_handle->user_data = user_data;
    }
}


void *vad_model_alloc(const char *model_path, void *user_data)
{
    struct ggml_handle_t *ggml_handle = ggml_model_alloc(model_path, 0, user_data);
    if(NULL == ggml_handle)
    {
        LOG_ERROR("asr ggml handle alloc error");
        return NULL;
    }
    if(SUCCESS != get_ggml_model(ggml_handle))
    {
        LOG_ERROR("no asr support model %s", ggml_handle->model_name);
        ggml_model_free(ggml_handle);
        return NULL;
    }

    ggml_handle->load_model(ggml_handle, model_path, 
                ggml_handle->input_shape, &ggml_handle->in_nodes, 
                ggml_handle->output_shape, &ggml_handle->out_nodes);

    ggml_data_alloc(ggml_handle, 
            ggml_handle->in_nodes, ggml_handle->input_shape, ggml_handle->input_matrix, 1,
            ggml_handle->out_nodes, ggml_handle->output_shape, ggml_handle->output_matrix, 1);

    return ggml_handle;
}

int vad_model_inference(void *handle, float *input_data, int input_size, float **output_data, int *output_size, int n_threads)
{
    struct ggml_handle_t *ggml_handle = (struct ggml_handle_t *)handle;
    memcpy(ggml_handle->input_matrix[1]->data_fp, input_data, input_size * sizeof(float));
    ggml_handle->inference(ggml_handle, ggml_handle->input_matrix, ggml_handle->output_matrix, n_threads, NULL);
    return SUCCESS;
}

void vad_model_free(void *handle)
{

}

#define GGML_VAD_TEST 0
#if GGML_VAD_TEST 
#include "audio_file.h"
int main(int argc, char *argv[])
{
    const char *model_path = "models/asr-fp32.gguf";
    const char *audio_path = "test/asr_example.wav";

    float *audio_data;
    uint32_t audio_size = 0;
    uint32_t sample_rate = 0;

    float *output_data = NULL;
    int output_size = 0;
    int offset = 0;

    void *handle = vad_model_alloc(model_path, NULL);

    load_audio_file(audio_path, &audio_data, &audio_size, &sample_rate);

    for(int i = 0; i < audio_size / 640; i++)
    {
        vad_model_inference(handle, &audio_data[offset], 640, &output_data, &output_size, 4);
    }

}
#endif
