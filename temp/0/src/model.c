#include "model.h"
#include "ggml_model.h"

void get_model(struct model_handle_t *model)
{
    if(NULL == model)
        return;
    memset(model, 0, sizeof(struct model_handle_t));
    model->init = init_ggml;
    model->deinit = deinit_ggml;
    model->alloc = ggml_model_alloc;
    model->data_alloc = ggml_data_alloc;
    model->data_free = ggml_data_free;
    model->free = ggml_model_free;
    model->inference = ggml_inference;
    model->use_gpu = ggml_use_gpu;
}


#if 0
int main(int argc, char *argv[])
{
    int in_nodes, out_nodes;
    static struct model_handle_t model;
    char *model_path = "/home/ysr/project/models/gguf/sense-voice-small-fp32.gguf";
    const char *wav_path = "example/asr_example.wav";
    float *wav_buf = NULL;
    int sample_width = sizeof(float);
    int sample_rate = 16000;
    int64_t wav_count = 0;
    int seg_sample = 512;
    int offset = 0;
    if(argc > 1)
    {   
        model_path = argv[1];
    }   

    wav_count = read_wav_file(wav_path, &wav_buf, 1, &sample_rate);
    if(ERROR == wav_count)
    {   
        LOG_ERROR("read_wav_file error %s ", wav_path);
        return ERROR;
    }   
    LOG_DEBUG("wav_path %s wav_count %d sample_width %d sampl_rate %d", wav_path, wav_count, sample_width, sample_rate);

    memset(&model, 0, sizeof(model));
    get_model(&model);
    model.init();

    //model.use_gpu(1); 

    model.handle = model.alloc(model_path, 0, model.input_shape, 
            &model.in_nodes, model.output_shape, &model.out_nodes, 6, NULL);

    if(NULL == model.handle)
    {   
        LOG_DEBUG("load retinaface model error");
        return ERROR;
    }   

    model.data_alloc(model.handle,
            model.in_nodes, model.input_shape, model.input_matrix, 1,
            model.out_nodes, model.output_shape,model.output_matrix, 1); 

    model.inference(model.handle, model.input_matrix, model.output_matrix, 0); 

#if 0
    while(1)
    {
    offset = 0;
    do{
        memcpy(model.input_matrix[1]->data_fp, wav_buf + offset, 512 * sizeof(float));
        model.inference(model.handle, model.input_matrix, model.output_matrix, 0); 
        offset += seg_sample;
        //usleep(seg_time);
    }while(wav_count > seg_sample + offset);
    }
#endif

    model.data_free(model.handle, model.input_matrix, model.output_matrix);
    model.free(model.handle);
    return SUCCESS;
}
#endif
