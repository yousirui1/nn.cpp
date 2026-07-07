#include "base.h"
#include "ggml_asr.h"
#include "ggml_model.h"
#include "ggml_sensevoice.h"
//#include "ggml_performer.h" to do

static struct ggml_model_t ggml_models[] =
{
    //name        load                inference           unload
    {"SenseVoiceSmall",  load_sensevoice_model, sensevoice_inference, NULL},
    {"SenseVoiceLarge",  load_sensevoice_model, sensevoice_inference, NULL},
    //{"performer", load_fsmn_model, fsmn_build_graph, fsmn_inference},
    //{"wenet",  load_silero_model, silero_build_graph, silero_inference},
};
static int get_ggml_model(struct ggml_handle_t *ggml_handle)
{
    int i;
    for(i = 0; i < ARRAY_SIZE(ggml_models); i++)
    {   
        if(STRPREFIX(ggml_models[i].name, ggml_handle->model_name))
        {
            LOG_DEBUG("find asr name %s", ggml_handle->model_name);
            ggml_handle->load_model = ggml_models[i].load_model;
            ggml_handle->inference = ggml_models[i].inference;
            //ggml_handle->unload_model = ggml_modules[i].unload_model;
            return SUCCESS;
        }
    }   
    return ERROR;
}


void *asr_model_alloc(const char *model_path, void *user_data)
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
            ggml_handle->in_nodes, ggml_handle->input_shape, ggml_handle->input_matrix, 0,
            ggml_handle->out_nodes, ggml_handle->output_shape, ggml_handle->output_matrix, 1);

    return ggml_handle;
}



int asr_model_inference(void *handle, float *input_data, int input_size, char **output_data, int *output_size, int n_thread)
{
    struct ggml_handle_t *ggml_handle = (struct ggml_handle_t *)handle;
    set_matrix_data(ggml_handle->input_matrix[0], input_data);
    ggml_handle->input_matrix[0]->shape.size = input_size;
    ggml_handle->inference(ggml_handle, ggml_handle->input_matrix, ggml_handle->output_matrix, n_thread, NULL);

    *output_data = (char *)&ggml_handle->output_matrix[0]->data_i8[0];
    *output_size = ggml_handle->output_matrix[0]->shape.dims[0];

    return SUCCESS;
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

    char *output_data = NULL;
    int output_size = 0;
    int offset = 0;

    void *handle = asr_model_alloc(model_path, NULL);

    load_audio_file(audio_path, &audio_data, &audio_size, &sample_rate);


    asr_model_inference(handle, audio_data, audio_size, &output_data, &output_size, 4);

    LOG_DEBUG("result size %d msg %s",  output_size, output_data);
    

}
#endif

