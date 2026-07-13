#include "base.h"
#include "tools.h"  //to do use ggml time
#include "gguf.h"
#include "ggml-backend.h"
#include "ggml_model.h"
#include "model_list.h"

static int is_init = 0;
static ggml_backend_t backend, cpu_backend;

static int get_ggml_model(struct ggml_handle_t *ggml_handle)
{
    int i;
    for(i = 0; i < ARRAY_SIZE(ggml_models); i++)
    {
        if(STRPREFIX(ggml_models[i].name, ggml_handle->model_name))
        {
            LOG_DEBUG("find name %s", ggml_handle->model_name);
            ggml_handle->load_model = ggml_models[i].load_model;
            ggml_handle->inference = ggml_models[i].inference;
            ggml_handle->unload_model = ggml_models[i].unload_model;
            return SUCCESS;
        }
    }
    return ERROR;
}

void print_tensor_shape(struct ggml_tensor *tensor)
{
    int i;
    printf("%s shape  [", tensor->name);
    for(i = 0; i < ggml_n_dims(tensor); i++)
    {
        printf("%d ", tensor->ne[i]);
    }
    printf("] ");

    printf("%s stride  [", tensor->name);
    for(i = 0; i < ggml_n_dims(tensor); i++)
    {
        printf("%d ", tensor->nb[i]);
    }
    printf("] ");
    //printf("]  view %p", tensor->view_src);
    printf(" TENSOR DATA TYPE %s \n", get_tensor_type_string(tensor->type));
}

void print_tensor(struct ggml_tensor *tensor)
{
#if 0
    if (!tensor || !tensor->data) {
        printf("tensor is NULL or has no data\n");
        return;
    }

    printf("tensor '%s': type=%s, shape=[%d, %d, %d, %d], nelements=%d\n",
        tensor->name,
        ggml_type_name(tensor->type),
        (int)tensor->ne[0], (int)tensor->ne[1],
        (int)tensor->ne[2], (int)tensor->ne[3],
        (int)ggml_nelements(tensor));

    float * data = (float *)tensor->data;
    int count = 0;
    FILE *fp = fopen("value.log", "w");

    for (int i3 = 0; i3 < tensor->ne[3]; i3++) {
        for (int i2 = 0; i2 < tensor->ne[2]; i2++) {
            for (int i1 = 0; i1 < tensor->ne[1]; i1++) {
                for (int i0 = 0; i0 < tensor->ne[0]; i0++) {
                    // 计算线性索引（考虑 stride）
                    size_t idx = i3 * tensor->nb[3] / sizeof(float)
                               + i2 * tensor->nb[2] / sizeof(float)
                               + i1 * tensor->nb[1] / sizeof(float)
                               + i0;

                    if(count % 10 == 0)
                    {
                        printf("\n line %d ", count / 10);
                        fprintf(fp, "\n line %d ", count / 10);
                    }

                    printf("%.2f ", data[idx]);
                    fprintf(fp, "%.2f ", data[idx]);
                    count ++;
                }
            }
        }
    }
#endif
}

int ggml_use_gpu(int enable)
{
#if 0
    struct ggml_handle_t *ggml_handle = (struct ggml_handle *)handle;
    if(NULL == ggml_handle)
        return ERROR;

    ggml_handle->use_gpu = enable;
#endif 
    //to do 
    return SUCCESS;
}


int init_ggml()
{
    if(is_init)
    {
        is_init ++;
        return SUCCESS;
    }

    is_init = 1;
    ggml_time_init();
    ggml_backend_load_all();
    backend = ggml_backend_init_best();

    if(GGML_BACKEND_DEVICE_TYPE_CPU != ggml_backend_dev_type(ggml_backend_get_device(backend)))
    {
        cpu_backend = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, NULL);
    }
    else
    {
        cpu_backend = backend;
    }
    //op
    return SUCCESS;
}

void deinit_ggml()
{
    if(1 == is_init)
    {
        if(cpu_backend)
        {
            if(cpu_backend != backend)
            {
                ggml_backend_free(cpu_backend);
            }
        }

        if(backend)
        {
            ggml_backend_free(backend);
        }
    
        backend = nullptr;
        cpu_backend = nullptr;
        is_init = 0;
    }
    else 
    {
        is_init--;
    }
}

int ggml_model_weight_alloc(ggml_context *ctx, ggml_backend_t dev_backend, ggml_backend_buffer_t &buffer, const std::map<std::string, ggml_tensor**> &tensors)
{
    auto buft = ggml_backend_get_default_buffer_type(dev_backend);
    auto alignment = ggml_backend_buft_get_alignment(buft);
    size_t mem_size = 0;

    for(const auto &[name, tensor] : tensors)
    {   
        mem_size += get_aligned_size(ggml_backend_buft_get_alloc_size(buft, *tensor), alignment);
    }   

    buffer = ggml_backend_buft_alloc_buffer(buft, mem_size);
    auto buffer_base = reinterpret_cast<char*>(ggml_backend_buffer_get_base(buffer));

    ggml_backend_buffer_set_usage(buffer, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);

    auto set_tensor = [&](ggml_tensor* tensor, const void* data, size_t size)
    {   
        ggml_backend_tensor_set_async(dev_backend, tensor, data, 0, size);
        buffer_base += get_aligned_size(ggml_backend_buft_get_alloc_size(buft, tensor), alignment);
    };  

    for (const auto& [name, tensor] : tensors)
    {
        size_t tensor_size = ggml_nbytes(*tensor);
        {
            auto new_tensor = ggml_new_tensor(ctx, (*tensor)->type, GGML_MAX_DIMS, (*tensor)->ne);
            ggml_backend_tensor_alloc(buffer, new_tensor, buffer_base);
            set_tensor(new_tensor, ggml_get_data(*tensor), tensor_size);
            *tensor = new_tensor;
        }
        ggml_set_param(*tensor);
        ggml_set_name(*tensor, name.c_str());
    }
    return mem_size;
}

void set_graph_backend(struct ggml_cgraph* gf, ggml_backend_sched_t sched, ggml_backend_t backend, ggml_backend_t cpu_backend, int nodes)
{
    if (nodes < 0)
        nodes = ggml_graph_n_nodes(gf);

    for (int i = 0; i != nodes; ++i)
    {
        struct ggml_tensor *node = ggml_graph_node(gf, i);
        if (node->op == GGML_OP_CUSTOM)
        {
            do {
                ggml_backend_sched_set_tensor_backend(sched, node, cpu_backend);
                if (++i == nodes)
                    return;
                node = ggml_graph_node(gf, i);
            } while ((node->op == GGML_OP_VIEW || node->op == GGML_OP_PERMUTE));
            goto set_main_backend;
        }
        else
        {
        set_main_backend:
            ggml_backend_sched_set_tensor_backend(sched, node, backend);
        }
    }
}

//to do
bool ggml_graph_compute_helper(ggml_backend_sched_t sched, struct ggml_cgraph *graph, int n_threads)
{
    for (int i = 0; i < ggml_backend_sched_get_n_backends(sched); ++i)
    {
        ggml_backend_t backend = ggml_backend_sched_get_backend(sched, i);
        ggml_backend_dev_t dev = ggml_backend_get_device(backend);
        ggml_backend_reg_t reg = dev ? ggml_backend_dev_backend_reg(dev) : NULL;

        ggml_backend_set_n_threads_t fn_set_n_threads = (ggml_backend_set_n_threads_t) ggml_backend_reg_get_proc_address(reg, "ggml_backend_set_n_threads");
        if (fn_set_n_threads)
        {
            fn_set_n_threads(backend, n_threads);
        }
    }
    bool t = ggml_backend_sched_graph_compute(sched, graph) == GGML_STATUS_SUCCESS;
    ggml_backend_sched_reset(sched);
    return t;
}

int ggml_data_alloc(void *handle,
        int in_nodes, shape_t *input_shape, matrix_t **input_matrix, int in_alloc_flag,                   int out_nodes, shape_t *output_shape, matrix_t **output_matrix, int out_alloc_flag)
{

    int i, j;
    struct ggml_handle_t *ggml_handle = (struct ggml_handle_t *)handle;

    for(i = 0; i < in_nodes; i++)
    {
        if(in_alloc_flag)
        {
            input_matrix[i] = matrix_alloc_shape(input_shape[i]);
        }
        else
        {
            input_matrix[i] = matrix_empty_shape(NULL, input_shape[i]);
        }
    }

    for(i = 0; i < out_nodes; i++)
    {
        if(out_alloc_flag)
        {
            output_matrix[i] = matrix_alloc_shape(output_shape[i]);
        }
        else
            output_matrix[i] = matrix_empty_shape(NULL, output_shape[i]);
    }

    if(out_alloc_flag == 2)
    {
        ggml_handle->is_dynamic = 1;
    }

    return SUCCESS;
}


int ggml_data_free(void *handle, matrix_t **input_matrix, matrix_t **output_matrix)
{
    int i;
    struct ggml_handle_t *ggml_handle = (struct ggml_handle_t *)handle;

    if(NULL == ggml_handle)
    {
        return ERROR;
    }

    for(i = 0; i < ggml_handle->in_nodes; i++)
    {
        if(input_matrix[i])
        {
            matrix_free(input_matrix[i]);
        }
        input_matrix[i] = NULL;
    }

    for(i = 0; i < ggml_handle->out_nodes; i++)
    {
        if(output_matrix[i])
            matrix_free(output_matrix[i]);
        output_matrix[i] = NULL;
    }
    return SUCCESS;
}


int ggml_inference(void *handle, matrix_t **input_matrix, matrix_t **output_matrix, int is_debug)
{
    struct ggml_handle_t *ggml_handle = (struct ggml_handle_t *)handle;
    uint64_t start_time, end_time;

    if(!ggml_handle || !ggml_handle->inference)
    {
	    LOG_ERROR("ggml_handle or inference is NULL error");
        return ERROR;
	}

    if(is_debug)
        start_time = get_ustime();

    ggml_handle->inference(ggml_handle, input_matrix, output_matrix);

    if(is_debug)
    {
        end_time = get_ustime();
        //LOG_DEBUG("once run device %s use %lld  ms ", device_name[ggml_handle->device_type],  (end_time - start_time)/ 1000);
        LOG_DEBUG("once run device use %lld  ms ", (end_time - start_time)/ 1000);
    }
    return SUCCESS;
}



void *ggml_model_alloc(const char *model_data, uint64_t model_size,
                        shape_t *input_shape, int *in_nodes,
                        shape_t *output_shape, int *out_nodes, int n_thread, void *user_data)

{
    int i;
    struct ggml_handle_t *ggml_handle = NULL;
    struct gguf_context *gguf_ctx = NULL;
    const char *model_type = NULL;
    struct ggml_context *file_ctx = NULL;

    ggml_handle = (struct ggml_handle_t *)malloc(sizeof(struct ggml_handle_t));
    if(NULL == ggml_handle)
    {
        LOG_ERROR("ggml handle malloc size %ld error %s", sizeof(struct ggml_handle_t), strerror(errno));        
        return NULL;
    }
    memset(ggml_handle, 0, sizeof(struct ggml_handle_t));

    struct gguf_init_params gguf_params = {
        /*.no_alloc  =*/ true,
        /*.ctx       =*/ &file_ctx,
    };

    ggml_handle->n_thread = n_thread;

    //to do gguf_init_from_file 释放问题
    gguf_ctx = gguf_init_from_file(model_data, gguf_params);
    if(!gguf_ctx)
    {
        LOG_ERROR("gguf init from file %s error", model_data);
        free(ggml_handle);
        return NULL;
    }

    LOG_DEBUG("version:      %d",gguf_get_version(gguf_ctx));
    LOG_DEBUG("alignment:   %zu",gguf_get_alignment(gguf_ctx));
    LOG_DEBUG("data offset: %zu",gguf_get_data_offset(gguf_ctx));
    LOG_DEBUG("n_kv : %zu",gguf_get_n_kv(gguf_ctx));

    model_type = gguf_get_val_str(gguf_ctx, 0);
    LOG_INFO("model_type %s", model_type);

    strncpy(ggml_handle->model_name, model_type, sizeof(ggml_handle->model_name));
    if(SUCCESS != get_ggml_model(ggml_handle))
    {
        LOG_ERROR("no support model %s", ggml_handle->model_name);
        free(ggml_handle);
        gguf_free(gguf_ctx);
        return NULL;
    }
    ggml_handle->backend = backend;
    ggml_handle->cpu_backend = cpu_backend;

    if(SUCCESS != ggml_handle->load_model(ggml_handle, model_data, model_size))
    {
        LOG_ERROR("ggml load model error");
        free(ggml_handle);
        return NULL;
    }
    printf("======================== ggml start ========================== \n");

    for(i = 0; i < ggml_handle->in_nodes; i++)
    {
        input_shape[i] = ggml_handle->input_shape[i];

        if(ggml_handle->input_names[i])
            strncpy(input_shape[i].name, ggml_handle->input_names[i], sizeof(input_shape[i].name));
        print_shape(input_shape[i]);
    }
    *in_nodes = ggml_handle->in_nodes;

    for(i = 0; i < ggml_handle->out_nodes; i++)
    {
        output_shape[i] = ggml_handle->output_shape[i];

        if(ggml_handle->output_names[i])
            strncpy(output_shape[i].name, ggml_handle->output_names[i], sizeof(output_shape[i].name));
        print_shape(output_shape[i]);
    }
    *out_nodes = ggml_handle->out_nodes;
    printf("======================== ggml end ========================== \n");

    gguf_free(gguf_ctx);
    return ggml_handle;
}

void ggml_model_free(void *handle)
{
    int i;
    struct ggml_handle_t *ggml_handle = (struct ggml_handle_t *)handle;

    if(!ggml_handle)
        return;

    if(ggml_handle->unload_model)
    {
        ggml_handle->unload_model(ggml_handle);
    }

    for(i = 0; i < ggml_handle->in_nodes; i++)
    {
        if(ggml_handle->input_names[i])
            free(ggml_handle->input_names[i]);

        ggml_handle->input_names[i] = NULL;
    }

    for(i = 0; i < ggml_handle->out_nodes; i++)
    {
        if(ggml_handle->output_names[i])
            free(ggml_handle->output_names[i]);

        ggml_handle->output_names[i] = NULL;
    }

    ggml_handle->load_model = NULL;
    ggml_handle->inference = NULL;
    ggml_handle->unload_model = NULL;
    free(ggml_handle);
}

int main(int argc, char *argv[])
{
    int in_nodes, out_nodes;
    shape_t input_shape[MAX_NUM_LAYER];
    shape_t output_shape[MAX_NUM_LAYER];

    matrix_t *input_matrix[MAX_NUM_LAYER];
    matrix_t *output_matrix[MAX_NUM_LAYER];

    void *handle = NULL;
    char *model_path = "/home/ysr/project/models/gguf/Fun-CosyVoice3-0.5B-2512-GGUF/CosyVoice3-2512_F32.gguf";
    if(argc > 1)
    {   
        model_path = argv[1];
    }   

    handle = ggml_model_alloc(model_path, 0, input_shape, &in_nodes, output_shape, &out_nodes, 1, NULL);

    ggml_data_alloc(handle, in_nodes, input_shape, input_matrix, 1,
                    out_nodes, output_shape, output_matrix, 1);


    ggml_inference(handle, input_matrix, output_matrix, 1);

    ggml_data_free(handle, input_matrix, output_matrix);
    ggml_model_free(handle);
    deinit_ggml();
    return SUCCESS;
}


#if 0
int main(int argc, char *argv[])
{
    int in_nodes, out_nodes;
    shape_t input_shape[MAX_NUM_LAYER];
    shape_t output_shape[MAX_NUM_LAYER];

    matrix_t *input_matrix[MAX_NUM_LAYER];
    matrix_t *output_matrix[MAX_NUM_LAYER];

    void *handle = NULL;
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
    init_ggml();

    handle = ggml_model_alloc(model_path, 0, input_shape, &in_nodes, output_shape, &out_nodes, 1, NULL);

    ggml_data_alloc(handle, in_nodes, input_shape, input_matrix, 1,
                    out_nodes, output_shape, output_matrix, 1);


    //ggml_inference(handle, input_matrix, output_matrix, 1);


    //while(1)
    {
    offset = 0;
    do{
        memcpy(input_matrix[1]->data_fp, wav_buf + offset, 512 * sizeof(float));
        ggml_inference(handle, input_matrix, output_matrix, 0); 
        offset += seg_sample;
        //usleep(seg_time);
    }while(wav_count > seg_sample + offset);
    }

    free(wav_buf);

    ggml_data_free(handle, input_matrix, output_matrix);
    ggml_model_free(handle);
    deinit_ggml();
    return SUCCESS;
}
#endif
