#include "base.h"
#include "gguf_loader.hpp"
#include "ggml_fsmn.hpp"
#include "ggml_handle.hpp"

void get_fsmn_default_params(struct fsmn_params_t *params)
{
    params->n_encoder_layer = 4;
}

ggml_cgraph *fsmn_build_cgraph(fsmn_model_t *model, fsmn_state_t *state, int chunk_size)
{
    struct ggml_init_params params = { 
            /*.mem_size   =*/state->meta.size(),
            /*.mem_buffer =*/state->meta.data(),
            /*.no_alloc   =*/true,
    };  

    struct ggml_context *ctx0 = ggml_init(params);

    ggml_cgraph *gf = ggml_new_graph_custom(ctx0, FSMN_MAX_NODES, false);

    ggml_tensor *speech = ggml_new_tensor_3d(ctx0, GGML_TYPE_F32, 1, chunk_size, 400);
    ggml_set_name(speech, "speech");
    ggml_set_input(speech);

    ggml_tensor *x;

    {
        x = model->in_linear1.build_cgraph(ctx0, speech);
        x = model->in_linear2.build_cgraph(ctx0, x);
    }

    {

    }

    {

    }

    ggml_set_output(x);
    ggml_build_forward_expand(gf, x);
    ggml_free(ctx0);
    return gf;
}

void unload_fsmn_model(struct ggml_handle_t *ggml_handle)
{

}

int load_fsmn_params(fsmn_params_t *params, gguf_loader &loader)
{
    std::string prefix = "frontend";

    int sample_rate;
    std::string window = "";
    int num_mels;
    int frame_length;
    int frame_shift;
    int lfr_m;
    int lfr_n;

    LOAD_METADATA(sample_rate);
    //LOAD_METADATA(window);
    LOAD_METADATA(num_mels);
    LOAD_METADATA(frame_length);
    LOAD_METADATA(frame_shift);
    LOAD_METADATA(lfr_m);
    LOAD_METADATA(lfr_n);

    params->sample_rate = sample_rate;
    params->window = window;
    params->num_mels = num_mels;
    params->frame_length = frame_length;
    params->frame_shift = frame_shift;
    params->lfr_m = lfr_m;
    params->lfr_n = lfr_n;

    return SUCCESS;
}

int load_fsmn_model(struct ggml_handle_t *ggml_handle, const char *model_path)
{
    struct ggml_context *ctx = nullptr;
    ggml_backend_buffer_t buffer = nullptr;

    struct fsmn_params_t *params = new fsmn_params_t;
    if(NULL == params)
    {   
        LOG_ERROR("fsmn params error %s", strerror(errno));
        return ERROR;
    }   
    get_fsmn_default_params(params);

    struct fsmn_model_t *model = new fsmn_model_t;
    if(NULL == model)
    {   
        LOG_ERROR("fsmn malloc error %s", strerror(errno));
        delete params;
        return ERROR;
    }   

    struct fsmn_state_t *state = new fsmn_state_t;
    if(NULL == model)
    {   
        LOG_ERROR("fsmn state error %s", strerror(errno));
        delete model;
        delete params;
        return ERROR;
    }

    gguf_loader loader(model_path);

    load_fsmn_params(params, loader);

    ctx = ggml_init(ggml_init_params{ .mem_size = ggml_graph_overhead() * GGML_DEFAULT_GRAPH_SIZE, .no_alloc = true });

    std::string prefix = "encoder";
    model->in_linear1.onload(loader, prefix + ".in_linear1.linear");
    model->in_linear2.onload(loader, prefix + ".in_linear2.linear");

    //to do
    model->encoders.resize(params->n_encoder_layer);
    for(int i = 0; i < params->n_encoder_layer; ++i)
    {   
        auto& layer = model->encoders[i];
        std::string name = prefix + ".fsmn."  + std::to_string(i);
        layer.onload(loader, name);
    }   

    model->out_linear1.onload(loader, prefix + ".out_linear1.linear");
    model->out_linear2.onload(loader, prefix + ".out_linear2.linear");

    return SUCCESS;
}

int fsmn_frontend_process(struct ggml_handle_t *ggml_handle, matrix_t **input_matrix)
{

    return SUCCESS;
}

int fsmn_backend_process(struct ggml_handle_t *ggml_handle, matrix_t **output_matrix)
{

    return SUCCESS;
}

int fsmn_inference(struct ggml_handle_t *ggml_handle, matrix_t **input_matrix, matrix_t **output_matrix)
{
#if 0
    struct silero_params_t *params = (struct silero_params_t *)ggml_handle->params;
    struct silero_model_t *model = (struct silero_model_t *)ggml_handle->model;
    struct silero_state_t *state = (struct silero_state_t *)ggml_handle->state;
#endif
    return SUCCESS;
}



