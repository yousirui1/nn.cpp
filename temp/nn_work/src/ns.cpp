


#include "base.h"
#if 0
#include "model.h"
#include "audio_tools.h"
#include "tools.h"

struct ns_handle_t
{
    int ns_level;
    float wav_cache[512];
};


matrix_t *ns_front_process(struct model_session_t *model_session, float *input_data, int input_size)
{   

}

const char *ns_back_process(struct model_session_t *model_session, void *output_data, int output_size)
{
    struct ns_handle_t *ns_handle = (struct ns_handle_t *)model_session->handle;
    struct model_handle_t *model = &model_session->model;
}


int ns_inference(struct model_session_t *model_session, void *input_data, int input_size, void **output_data, int *output_size, void *param)
{
    int sample_width = 4; 
    float *input_data_fp = NULL;
    struct model_handle_t *model = &model_session->model;
    matrix_t *feature = NULL;
    const char *text = NULL;
    if(param)
    {
        sample_width = *(int *)param;
    }
    LOG_DEBUG("sample_width %d", sample_width);
    
    if(sample_width == sizeof(int16_t))
    {
        input_data_fp = int16_to_float_data(input_data, input_size); //to do 
    }   
    else
    {
        input_data_fp = (float *)input_data;
    }
    
    if(NULL == input_data_fp)
    {   
        LOG_ERROR("input_data is NULL error");
        return ERROR;
    }

    set_shape(&model->input_matrix[0]->shape, 2, 1, input_size);
    model->input_matrix[0]->data = input_data_fp;
    model->inference(model->handle, model->input_matrix, model->output_matrix, 1);

    if(output_data)
        *output_data = model->output_matrix[0]->data_fp;
        *output_size = model->output_matrix[0]->shape.size;
    
    //if(model_session->cb)
    //   model_session->cb(text, strlen(text), model_session->user_data);
    return SUCCESS;
}


int ns_stream_inference(struct model_session_t *model_session, void *input_data, int input_size, void *param)
{
    int sample_width = 4;
    float *input_data_fp = NULL;
    struct model_handle_t *model = &model_session->model;
    
    if(param)
    {
        sample_width = *(int *)param;
    }   
    
    if(sample_width == sizeof(int16_t))
    {   
        input_data_fp = int16_to_float_data((int16_t *)input_data, input_size);
    }
    else
    {
        input_data_fp = (float *)input_data;
    }   

    /* conv cache */
    memcpy(model->input_matrix[2]->data,  model->output_matrix[1]->data,
                model->input_matrix[2]->shape.size * sizeof(float));

    /* tra cache */
    memcpy(model->input_matrix[3]->data,  model->output_matrix[2]->data,
                model->input_matrix[3]->shape.size * sizeof(float));

    /* inter cache */
    memcpy(model->input_matrix[4]->data,  model->output_matrix[3]->data,
                model->input_matrix[4]->shape.size * sizeof(float));

    /* istft cache */
    memcpy(model->input_matrix[5]->data,  model->output_matrix[4]->data,
                model->input_matrix[5]->shape.size * sizeof(float));


    model->input_matrix[0]->data = input_data_fp;
    model->input_matrix[0]->data_fp = input_data_fp;
    model->inference(model->handle, model->input_matrix, model->output_matrix, 1);


    //print_matrix(model->output_matrix[0], "output");

    /* wave cache */
    memcpy(model->input_matrix[1]->data,  input_data_fp,
                model->input_matrix[1]->shape.size * sizeof(float));

    //if(output_data)
    //    *output_data = model->output_matrix[0]->data_fp;
    //    *output_size = model->output_matrix[0]->shape.size;
    
    if(model_session->cb)
       model_session->cb(model->output_matrix[0]->data_fp, model->output_matrix[0]->shape.size, model_session->user_data);
 

#if 0
    //to do 
    if(model_session->is_running && input_data_fp)
    {
        en_queue(&model_session->queue, input_data_fp, input_size * sizeof(float), 0);
    }
#endif
}


static void ns_loop(struct model_session_t *model_session, int timeout)
{
    int sample_width = sizeof(float);  
    struct queue_index_t *index = NULL;
    for(;;)
    {
        if(!model_session->is_running)
        {   
            break;
        }

        if(empty_queue(&model_session->queue))
        {   
            usleep(timeout);
            continue;
        }
        
        index = de_queue(&model_session->queue); 
        ns_inference(model_session, index->buf, index->size/sizeof(float), NULL, 0, &sample_width);
        de_queue_pos(&model_session->queue);
    }
}


EXPORT void ns_register_callback(struct model_session_t *model_session, void *cb, void *user_data)
{
    model_session->cb = cb;
    model_session->user_data = user_data;
}

void deinit_ns_model(struct model_session_t *model_session)
{
    struct model_handle_t *model = &model_session->model;
    model->data_free(model->handle, model->input_matrix, model->output_matrix);
    model->free(model->handle);
    model->deinit();
}


int init_ns_model(struct model_session_t *model_session, const char *model_path, int cache_size)
{
    struct model_handle_t *model = &model_session->model;
    struct ns_handle_t *ns_handle = (struct ns_handle_t *)model_session->handle;

    get_model(model);

    model->init();
    model->handle = model->alloc(model_path, 0, model->input_shape,
            &model->in_nodes, model->output_shape, &model->out_nodes, NULL);

    if(NULL == model->handle)
    {
        LOG_DEBUG("load retinaface model error");
        return ERROR;
    }

    model->data_alloc(model->handle,
            model->in_nodes, model->input_shape, model->input_matrix, 0,
            model->out_nodes, model->output_shape,model->output_matrix, 2);

    if(cache_size > 0)
    {
        model_session->queue_buf = (uint8_t *)malloc(cache_size);
        if(NULL == model_session->queue_buf)
        {
            LOG_ERROR("queue_buf malloc size %d error %s", cache_size, strerror(errno));
            deinit_ns_model(model_session);
            return ERROR;
        }
        LOG_DEBUG("cache_size %d", cache_size);
        init_queue(&model_session->queue, model_session->queue_buf, cache_size);
    }
    return SUCCESS;
}

int init_ns_stream_model(struct model_session_t *model_session, const char *model_path, int cache_size)
{
    struct model_handle_t *model = &model_session->model;
    struct ns_handle_t *ns_handle = (struct ns_handle_t *)model_session->handle;

    get_model(model);

    model->init();
    model->handle = model->alloc(model_path, 0, model->input_shape,
            &model->in_nodes, model->output_shape, &model->out_nodes, NULL);

    if(NULL == model->handle)
    {
        LOG_DEBUG("load retinaface model error");
        return ERROR;
    }

    model->data_alloc(model->handle,
            model->in_nodes, model->input_shape, model->input_matrix, 1,
            model->out_nodes, model->output_shape,model->output_matrix, 1);

    if(cache_size > 0)
    {
        model_session->queue_buf = (uint8_t *)malloc(cache_size);
        if(NULL == model_session->queue_buf)
        {
            LOG_ERROR("queue_buf malloc size %d error %s", cache_size, strerror(errno));
            deinit_ns_model(model_session);
            return ERROR;
        }
        LOG_DEBUG("cache_size %d", cache_size);
        init_queue(&model_session->queue, model_session->queue_buf, cache_size);
    }
    return SUCCESS;
}

//to do level
struct model_session_t *ns_model_alloc(int ns_level) 
{
    struct model_session_t *model_session = NULL;

    struct ns_handle_t *ns_handle = (struct ns_handle_t *)malloc(sizeof(struct ns_handle_t));
    if(NULL == ns_handle)
    {
        LOG_ERROR("ns_handle malloc size %ld error %s ", sizeof(struct ns_handle_t), strerror(errno));
        return NULL;
    }
    memset(ns_handle, 0, sizeof(struct ns_handle_t));

    ns_handle->ns_level = ns_level;
    model_session = (struct model_session_t *)malloc(sizeof(struct model_session_t));
    memset(model_session, 0, sizeof(struct model_session_t));

    model_session->handle = ns_handle;
    return model_session;
}


EXPORT void ns_model_free(struct model_session_t *model_session)
{
    struct ns_handle_t *ns_handle = NULL;
    if(NULL == model_session)
        return;

    ns_handle = (struct ns_handle_t *)model_session->handle;

    if(NULL == ns_handle)
        return;

    model_session->handle = NULL;

    if(model_session->queue_buf)
        free(model_session->queue_buf);

    model_session->queue_buf = NULL;
    model_session->cb = NULL;
    model_session->user_data = NULL;

    free(ns_handle);
    free(model_session);
}


static void *ns_thread(void *param)
{
    int ret;
    int timeout = 10000;
    pthread_attr_t st_attr;
    struct sched_param sched;
    struct ns_model_t *ns_model = param;
    struct model_session_t *model_session = (struct model_session_t *)param;

    //pthread_detach(pthread_self());
    ret = pthread_attr_init(&st_attr);
    if(ret)
    {
        LOG_DEBUG("thread server tcp attr init warning ");
    }
    ret = pthread_attr_setschedpolicy(&st_attr, SCHED_FIFO);
    if(ret)
    {
        LOG_DEBUG("thread server tcp set SCHED_FIFO warning");
    }
    sched.sched_priority = 0;
    ret = pthread_attr_setschedparam(&st_attr, &sched);

    pthread_setcancelstate(PTHREAD_CANCEL_ENABLE, NULL);     //线程可以被取消掉
    pthread_setcanceltype(PTHREAD_CANCEL_ASYNCHRONOUS,NULL); //立即退出
    //pthread_setcanceltype(PTHREAD_CANCEL_DEFERRED, NULL);  //立即退出  PTHREAD_CANCEL_DEFERRED

    model_session->is_running = 1;

    ns_loop(model_session, timeout);
    return (void *)SUCCESS;
}


EXPORT int start_ns_thread(struct model_session_t *model_session, int timeout)
{
    int ret;
    ret = pthread_create(&model_session->tid, NULL, ns_thread, model_session);
    if(ret != SUCCESS)
    {
        LOG_ERROR("start ns inference thread error %s ", strerror(errno));
        return ERROR;
    }
    LOG_DEBUG("start ns inference thread end");
    return SUCCESS;
}

EXPORT void stop_ns_thread(struct model_session_t *model_session)
{
    void *tret = NULL;
    model_session->is_running = 1;
    pthread_join(model_session->tid, &tret);
    LOG_DEBUG("stop ns inference thread end");
}


#define NS_TEST 1
#if NS_TEST
static void ns_callback(const float *output_buf, int output_size, void *user_data)
{
#if 0
    char path[128] = {0};
    static int count = 0;
    snprintf(path, sizeof(path), "output/test_%d.wav", count++);
    //ns_model_nonblock_process(user_data, data, size);
    LOG_DEBUG("callback %d", size);
    write_wav_file(path, data, size, 16000, 1);
#endif
    static FILE *fp = NULL;

    if(fp == NULL)
    {
        fp = fopen("output.pcm", "wb");
    }

    //LOG_DEBUG("ns result size %d", output_size);
    fwrite(output_buf, output_size, sizeof(float), fp);
}

int main(int argc, char *argv[])
{
#if 0
    int ns_level = 2;
    // quantize 损失比较大推理时间没减少
    const char *model_path = "/home/ysr/project/ai/open_source/gtcrn/onnx_models/gtcrn_simple.onnx";
    const char *stream_model_path = "/home/ysr/project/ai/open_source/gtcrn/onnx_models/gtcrn_stream.onnx";
    const char *wav_path = "/home/ysr/project/ai/open_source/gtcrn/test.wav";
    void *wav_buf = NULL;
    int64_t wav_count = 0;
    int sample_width = 0;
    int sample_rate = 0;
    float *output_buf = NULL;
    int output_size = 0;

    struct model_session_t *model_session = ns_model_alloc(ns_level);
    init_ns_model(model_session, model_path, 0);
    ns_register_callback(model_session, ns_callback, NULL);

    wav_count = read_wav_file(wav_path, &wav_buf, 1, &sample_width, &sample_rate);
    if(ERROR == wav_count)
    {
        LOG_ERROR("read_wav_file error %s ", wav_path);
        return ERROR;
    }
    LOG_DEBUG("wav_path %s wav_count %d sample_width %d sampl_rate %d", wav_path, wav_count, sample_width, sample_rate);

    ns_inference(model_session, wav_buf, wav_count, &output_buf, &output_size, &sample_width);
    
    write_wav_file("output.wav", output_buf, output_size, sizeof(float), 16000);

    deinit_ns_model(model_session);
    ns_model_free(model_session);
#else
    int ns_level = 2;
    // quantize 损失比较大推理时间没减少
    const char *model_path = "/home/ysr/project/ai/open_source/gtcrn/onnx_models/gtcrn_simple.onnx";
    //const char *stream_model_path = "/home/ysr/project/ai/open_source/gtcrn/onnx_models/gtcrn_stream.onnx";
    const char *stream_model_path = "/home/ysr/project/ai/open_source/gtcrn/onnx_models/gtcrn_stream_simple_quantize.onnx";
    const char *wav_path = "/home/ysr/project/ai/open_source/gtcrn/test.wav";
    void *wav_buf = NULL;
    int64_t wav_count = 0;
    int sample_width = 0;
    int sample_rate = 0;
    float *output_buf = NULL;
    int output_size = 0;
    int offset = 0;
    int seg_sample = 256;       //to do 256 
    float *wav_buf_fp = NULL;
    int16_t *wav_buf_i16 = NULL;


    struct model_session_t *model_session = ns_model_alloc(ns_level);
    init_ns_stream_model(model_session, stream_model_path, 0);
    ns_register_callback(model_session, ns_callback, NULL);

    wav_count = read_wav_file(wav_path, &wav_buf, 1, &sample_width, &sample_rate);
    if(ERROR == wav_count)
    {
        LOG_ERROR("read_wav_file error %s ", wav_path);
        return ERROR;
    }
    LOG_DEBUG("wav_path %s wav_count %d sample_width %d sampl_rate %d", wav_path, wav_count, sample_width, sample_rate);

    wav_buf_i16 = (int16_t *)wav_buf;
    wav_buf_fp = (float *)wav_buf;

    do{
        if(sample_width == sizeof(int16_t))
            ns_stream_inference(model_session, &wav_buf_i16[offset], seg_sample, &sample_width);
        else 
            ns_stream_inference(model_session, &wav_buf_fp[offset], seg_sample, &sample_width);

        offset += seg_sample;
    }while(wav_count > seg_sample + offset);

    deinit_ns_model(model_session);
    ns_model_free(model_session);
    free(wav_buf);
#endif
}

#endif // NS_TEST


#endif
