#include "base.h"
#include "ggml_tts.h"
#include "ggml_model.h"
#include "ggml_cosyvoice.h"
//#include "ggml_performer.h" to do

static struct ggml_model_t ggml_models[] =
{
    //name        load                inference           unload
    {"cosyvoice3-2512",  load_cosyvoice_model, cosyvoice_inference, NULL},
    //{"cosyvoice2",  load_sensevoice_model, sensevoice_inference, NULL},
    //{"kantts",  load_sensevoice_model, sensevoice_inference, NULL},
    //{"indexTTS",  load_sensevoice_model, sensevoice_inference, NULL},
};

static int get_ggml_model(struct ggml_handle_t *ggml_handle)
{
    int i;
    for(i = 0; i < ARRAY_SIZE(ggml_models); i++)
    {
        if(STRPREFIX(ggml_models[i].name, ggml_handle->model_name))
        {
            LOG_DEBUG("find tts model  name %s", ggml_handle->model_name);
            ggml_handle->load_model = ggml_models[i].load_model;
            ggml_handle->inference = ggml_models[i].inference;
            //ggml_handle->unload_model = ggml_modules[i].unload_model;
            return SUCCESS;
        }
    }
    return ERROR;
}

void *tts_model_alloc(const char *model_path, void *user_data)
{
    struct ggml_handle_t *ggml_handle = ggml_model_alloc(model_path, 0, user_data);
    if(NULL == ggml_handle)
    {
        LOG_ERROR("asr ggml handle alloc error");
        return NULL;
    }

    if(SUCCESS != get_ggml_model(ggml_handle))
    {   
        LOG_ERROR("no tts support model %s", ggml_handle->model_name);
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

int tts_model_inference(void *handle,  char *input_data, int input_size, float **output_data, int *output_size, int n_thread)
{
    struct ggml_handle_t *ggml_handle = (struct ggml_handle_t *)handle;

    set_matrix_data(ggml_handle->input_matrix[0], input_data);
    ggml_handle->input_matrix[0]->shape.size = input_size;
    ggml_handle->inference(ggml_handle, ggml_handle->input_matrix, ggml_handle->output_matrix, n_thread, NULL);

    *output_data = &ggml_handle->output_matrix[0]->data_fp[0];
    *output_size = ggml_handle->output_matrix[0]->shape.dims[0];
    return SUCCESS;
}

void tts_model_free(void *handle)
{
    //t odo 

}
