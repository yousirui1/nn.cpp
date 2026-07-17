#include "base.h"
#include <functional>
#include "ggml_model.h"
#include "gguf.h"

//static int device_type = USE_GPU;

inline static const char *get_tensor_type_string(int type)
{
    switch (type)
    {
        case GGML_TYPE_F16:
            return "FP16";
        case GGML_TYPE_F32:
            return "FP32";
        case GGML_TYPE_I8:
            return "INT8";
        case GGML_TYPE_I16:
            return "INT16";
        case GGML_TYPE_I32:
            return "INT32";
        case GGML_TYPE_I64:
            return "INT64";
        default:
            return "Q_K";
    }
}

//batch_size to do
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
    printf("]  view %p", tensor->view_src);

    printf(" TENSOR DATA TYPE %s \n", get_tensor_type_string(tensor->type));
}


void ggml_print_tensor(struct ggml_tensor * tensor) {
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

    // 获取原始数据指针
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
}



void set_graph_backend(ggml_cgraph* gf, ggml_backend_sched_t sched, ggml_backend_t backend, ggml_backend_t cpu_backend, int nodes)
{
    if (nodes < 0)
        nodes = ggml_graph_n_nodes(gf);

    for (int i = 0; i != nodes; ++i) 
    {
        auto node = ggml_graph_node(gf, i);
        if (node->op == GGML_OP_CUSTOM)
        {
            do {
                ggml_backend_sched_set_tensor_backend(sched, node, cpu_backend);
                if (++i == nodes) return;
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

bool ggml_graph_compute_helper(ggml_backend_sched_t sched, struct ggml_cgraph *graph, int n_threads) 
{
    for (int i = 0; i < ggml_backend_sched_get_n_backends(sched); ++i) 
    {
        ggml_backend_t backend = ggml_backend_sched_get_backend(sched, i);
        ggml_backend_dev_t dev = ggml_backend_get_device(backend);
        ggml_backend_reg_t reg = dev ? ggml_backend_dev_backend_reg(dev) : nullptr;

        auto * fn_set_n_threads = (ggml_backend_set_n_threads_t) ggml_backend_reg_get_proc_address(reg, "ggml_backend_set_n_threads");
        if (fn_set_n_threads) 
        {
            fn_set_n_threads(backend, n_threads);
        }
    }
    bool t = ggml_backend_sched_graph_compute(sched, graph) == GGML_STATUS_SUCCESS;
    ggml_backend_sched_reset(sched);
    return t;
}

struct ggml_handle_t *ggml_model_alloc(const char *model_data, uint64_t model_size, void *user_data)
{
    int i, j;
    int thread_num = 1;
    const char *name = NULL;
    struct ggml_tensor *cur = NULL;
    size_t num_bytes = 0;
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

    ggml_handle->n_thread = thread_num;

    gguf_ctx = gguf_init_from_file(model_data, gguf_params);

    LOG_INFO("version:      %d",gguf_get_version(gguf_ctx));
    LOG_INFO("alignment:   %zu",gguf_get_alignment(gguf_ctx));
    LOG_INFO("data offset: %zu",gguf_get_data_offset(gguf_ctx));
    LOG_INFO("n_kv : %zu",gguf_get_n_kv(gguf_ctx));

    model_type = gguf_get_val_str(gguf_ctx, 0);
    LOG_DEBUG("model_type %s", model_type);

    strncpy(ggml_handle->model_name, model_type, sizeof(ggml_handle->model_name));

    ggml_handle->backend = ggml_backend_init_best();

    if(GGML_BACKEND_DEVICE_TYPE_CPU != ggml_backend_dev_type(ggml_backend_get_device(ggml_handle->backend)))
    {
        ggml_handle->cpu_backend = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, NULL);
    }
    else
    {
        ggml_handle->cpu_backend = ggml_handle->backend;    
    }
    //to do gpu


    struct ggml_init_params params = {
        ggml_graph_overhead(),
        nullptr,
        true,
    };

    struct ggml_context *ctx = ggml_init(params);
    if(ctx)
    {
        struct ggml_tensor *a = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, 1);
        a = ggml_concat(ctx, a, a, 0);
        ggml_handle->op_caps.concat_i32 = ggml_backend_supports_op(ggml_handle->backend, a);

        a = ggml_new_tensor_3d(ctx, GGML_TYPE_F16, 11, 4, 51);
        a = ggml_repeat_4d(ctx, a, 11, 4, 51, 4);
        ggml_handle->op_caps.repeat_f16 = ggml_backend_supports_op(ggml_handle->backend, a);

        ggml_free(ctx);
        LOG_DEBUG("concat_i32 %d repeat_f16 %d", ggml_handle->op_caps.concat_i32, ggml_handle->op_caps.repeat_f16);
    }
    ggml_handle->user_data = user_data;

    return ggml_handle;    
}

int ggml_data_alloc(struct ggml_handle_t * ggml_handle,
         int in_nodes, shape_t *input_shape, matrix_t **input_matrix, int in_alloc_flag,
         int out_nodes, shape_t *output_shape, matrix_t **output_matrix, int out_alloc_flag)
{
    int i, j;

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



void ggml_data_free()
{

    //to do 
}

void ggml_model_free(struct ggml_handle_t *ggml_handle)
{
    //t odo 
}

#if 0
int get_ggml_model(struct ggml_handle_t *ggml_handle, struct ggml_model_t ggml_models[])
{
    int i;
    LOG_DEBUG("ARRAY_SIZE(ggml_models) %d ", ARRAY_SIZE(ggml_models));
    for(i = 0; i < ARRAY_SIZE(ggml_models); i++)
    {
        LOG_DEBUG("ggml_models[i].name %s ggml_handle->model_name) %s", ggml_models[i].name, ggml_handle->model_name);
        if(STRPREFIX(ggml_models[i].name, ggml_handle->model_name))
        {
            LOG_DEBUG("name %s", ggml_handle->model_name);
            ggml_handle->load_model = ggml_models[i].load_model;
            ggml_handle->inference = ggml_models[i].inference;
            //ggml_handle->unload_model = ggml_modules[i].unload_model;
            return SUCCESS;
        }
    }
    return ERROR;
}
#endif

