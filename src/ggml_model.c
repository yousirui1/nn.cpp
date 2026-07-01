#include "base.h"
#include "model.h"
#include "gguf.h"
#include "ggml_model.h"

struct ggml_model_t
{
    char name[128];
   
    int (*load_model)(struct ggml_handle_t *ggml_handle, const char *model_path);

    int (*inference)(struct ggml_handle_t *ggml_handle, matrix_t **input_matrix, matrix_t **output_matrix);
    void (*unload_model)(struct ggml_handle_t *ggml_handle);
};

static struct ggml_model_t ggml_models[] = 
{
    //name                  load        inference       unload
    {"SenseVoiceSmall",     NULL,       NULL ,        NULL},
    {"FsmnVad",             NULL,       NULL ,        NULL},
};

static int is_ggml_init = 0;
static ggml_backend_t ggml_backend;

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


int ggml_use_gpu(void *handle, int enable)
{
    struct ggml_handle_t *ggml_handle = (struct ggml_handle *)handle;
    if(NULL == ggml_handle)
        return ERROR;

    ggml_handle->use_gpu = enable;
    return SUCCESS;
}

int init_ggml()
{
    if(is_ggml_init)
    {
        is_ggml_init++;
        return SUCCESS;
    }
    ggml_time_init();
    ggml_backend_load_all();
    ggml_backend = ggml_backend_init_best();
    
#if 0
    struct ggml_init_params params = {
       /*.mem_size   =*/ .mem_size = ggml_graph_overhead(), 
       /*.no_alloc   =*/ .no_alloc = true,
    };

    struct ggml_context *ctx = ggml_init(params);
    if(ctx)
    {

    ggml_tensor* a = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, 1); 
    a = ggml_concat(ctx0, a, a, 0); 
    op_caps.concat_i32 = ggml_backend_supports_op(backend, a); 

    a = ggml_new_tensor_3d(ctx0, GGML_TYPE_F16, 11, 4, 51);
    a = ggml_repeat_4d(ctx0, a, 11, 4, 51, 4); 
    op_caps.repeat_f16 = ggml_backend_supports_op(backend, a); 

    a = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, 16, 4); 
    a = ggml_pad(ctx0, a, 4, 0, 0, 0); 
    op_caps.pad = ggml_backend_supports_op(backend, a); 

    a = ggml_fill(ctx0, a, 0.0f);
    op_caps.fill = ggml_backend_supports_op(backend, a); 

    a = ggml_cumsum(ctx0, a); 
    op_caps.cumsum = ggml_backend_supports_op(backend, a); 

    a = ggml_new_tensor_1d(ctx0, GGML_TYPE_F32, 16);
    a = ggml_pad_reflect_1d(ctx0, a, 1, 0); 
    op_caps.pad_reflect_1d = ggml_backend_supports_op(backend, a); 

    op_caps.top_k = ggml_backend_supports_op(backend, ggml_top_k(ctx0, a, 5));

    op_caps.leaky_relu = ggml_backend_supports_op(backend, ggml_leaky_relu(ctx0, a, 0.01f, false));

    op_caps.sin = ggml_backend_supports_op(backend, ggml_sin(ctx0, a));
    op_caps.cos = ggml_backend_supports_op(backend, ggml_cos(ctx0, a));

    op_caps.arange = ggml_backend_supports_op(backend, ggml_arange(ctx0, 0.f, 10.f, 1.f));

    op_caps.elu = ggml_backend_supports_op(backend, ggml_elu(ctx0, a));
    op_caps.abs = ggml_backend_supports_op(backend, ggml_abs(ctx0, a));
    op_caps.floor = ggml_backend_supports_op(backend, ggml_floor(ctx0, a));

    {   
        auto a2 = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, 16, 4); 
        auto b = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, 16, 4); 
        op_caps.acc = ggml_backend_supports_op(backend, ggml_acc(ctx0, a2, b, a2->nb[1], a2->nb[2], a2->nb[3], 0));
    }   

    {   
        ggml_tensor* w = ggml_new_tensor_3d(ctx0, GGML_TYPE_F16, 3, 4, 8); 
        a = ggml_new_tensor_3d(ctx0, GGML_TYPE_F16, 16, 4, 1);
        a = ggml_im2col(ctx0, w, a, 1, 0, 0, 0, 1, 0, false, GGML_TYPE_F16);
        op_caps.im2col_f16 = ggml_backend_supports_op(backend, a);
    }
        struct ggml_tensor *a = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, 1);
        a = ggml_concat(ctx, a, a, 0);
        ggml_handle->op_caps.concat_i32 = ggml_backend_supports_op(ggml_handle->backend, a);

        a = ggml_new_tensor_3d(ctx, GGML_TYPE_F16, 11, 4, 51);
        a = ggml_repeat_4d(ctx, a, 11, 4, 51, 4);
        ggml_handle->op_caps.repeat_f16 = ggml_backend_supports_op(ggml_handle->backend, a);

        ggml_free(ctx);
        LOG_DEBUG("concat_i32 %d repeat_f16 %d", ggml_handle->op_caps.concat_i32, ggml_handle->op_caps.repeat_f16);
    }
#endif
    is_ggml_init = 1;
    return SUCCESS;
}

//to do
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

void deinit_ggml()
{
    if(is_ggml_init == 1)
    {
        is_ggml_init = 0;
    }
    else
    {
        is_ggml_init --;
    }
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

    gguf_ctx = gguf_init_from_file(model_data, gguf_params);

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
        return NULL;
    }

    ggml_handle->backend = ggml_backend;
    if(GGML_BACKEND_DEVICE_TYPE_CPU != ggml_backend_dev_type(ggml_backend_get_device(ggml_handle->backend)))
    {
        ggml_handle->cpu_backend = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, NULL);
    }
    else
    {
        ggml_handle->cpu_backend = ggml_handle->backend;
    }


    if(SUCCESS != ggml_handle->load_model(ggml_handle, model_data))
    {
        LOG_ERROR("ggml load model error");
        free(ggml_handle);
        return NULL;
    }
    printf("======================== ggml start ========================== \n");

    for(i = 0; i < ggml_handle->in_nodes; i++)
    {   
        //to do 
        input_shape[i] = ggml_handle->input_shape[i];

        if(ggml_handle->input_names[i])
            strncpy(input_shape[i].name, ggml_handle->input_names[i], sizeof(input_shape[i].name));
        print_shape(input_shape[i]);
    }
    
    for(i = 0; i < ggml_handle->out_nodes; i++)
    {
        //to do 
        output_shape[i] = ggml_handle->output_shape[i];

        if(ggml_handle->output_names[i])
            strncpy(output_shape[i].name, ggml_handle->output_names[i], sizeof(output_shape[i].name));
        print_shape(output_shape[i]);
    }
    printf("======================== ggml end ========================== \n");
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
        
        //if(ggml_handle->input_matrix[i])
        //    matrix_free(ggml_handle->input_matrix[i]);

        ggml_handle->input_names[i] = NULL;
        //ggml_handle->input_matrix[i] = NULL;
    }

    for(i = 0; i < ggml_handle->out_nodes; i++)
    {   
        if(ggml_handle->output_names[i])
            free(ggml_handle->output_names[i]);
        
        //if(ggml_handle->output_matrix[i])
        //    matrix_free(ggml_handle->output_matrix[i]);

        ggml_handle->output_names[i] = NULL;
        //ggml_handle->output_matrix[i] = NULL;
    }

    ggml_handle->load_model = NULL;
    ggml_handle->inference = NULL;
    ggml_handle->unload_model = NULL;
    //to do 

    free(ggml_handle);
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

    if(ggml_handle && ggml_handle->inference)
    {
        if(is_debug)
            start_time = get_ustime();

        ggml_handle->inference(ggml_handle, input_matrix, output_matrix);

        if(is_debug)
        {
            end_time = get_ustime();
            LOG_DEBUG("once run device %s use %lld  ms ", device_name[ggml_handle->device_type],  (end_time - start_time)/ 1000);
        }
    }
}

