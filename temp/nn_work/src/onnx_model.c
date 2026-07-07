#include "base.h"


#if 0

#include "onnx_model.h"

static OrtEnv *env = NULL;

//mutli model
static OrtApi *ort = NULL;
static OrtMemoryInfo* memory_info = NULL;
static int is_init = 0;
static int device_type = USE_CPU;

static int onnx_enable_openvino(OrtSessionOptions* session_options)
{
    OrtOpenVINOProviderOptions o;
    memset(&o, 0, sizeof(o));
    o.device_type = "GPU"; // CPU AVX512
    o.device_id = 0;

    OrtStatus* onnx_status = ort->SessionOptionsAppendExecutionProvider_OpenVINO(&session_options, &o);
    if(onnx_status != NULL)
    {
        const char* msg = ort->GetErrorMessage(onnx_status);
        LOG_ERROR("open openvino error %s", msg);
        ort->ReleaseStatus(onnx_status);
        return ERROR;
    }
    LOG_DEBUG("use openvino success");
    return SUCCESS;
}

static int onnx_enable_cuda(OrtSessionOptions* session_options)
{
    OrtCUDAProviderOptions o;
    memset(&o, 0, sizeof(o));
    o.device_id = 0;
    //o.gpu_mem_limit = SIZE_MAX;

    o.do_copy_in_default_stream = 1;
    //o.cudnn_conv_algo_search = OrtCudnnConvAlgoSearchHeuristic;
    //o.arena_extend_strategy = 1;


    OrtStatus* onnx_status = ort->SessionOptionsAppendExecutionProvider_CUDA(session_options, &o);
    if(onnx_status != NULL)
    {
        const char* msg = ort->GetErrorMessage(onnx_status);
        LOG_ERROR("open openvino error %s", msg);
        ort->ReleaseStatus(onnx_status);
        return ERROR;
    }
    //to do 只是声明内存已经在GPU上了，指针是GPU指针,所以创建的时候需要用GPU malloc 
    // cudaMalloc(&gpu_input, size); cudaMemcpy(gpu_input, cpu_input, size, cudaMemcpyHostToDevice);
 
    LOG_DEBUG("use cuda success");
    return SUCCESS;
}

static int onnx_enable_tensorRT(OrtSessionOptions* session_options)
{
    OrtTensorRTProviderOptions o;
    memset(&o, 0, sizeof(o));
    o.device_id = 0;
    o.trt_max_workspace_size = 1<<30;
    //o.trt_fp16_enable = 1;

    //o.do_copy_in_default_stream = 1;

    OrtStatus* onnx_status = ort->SessionOptionsAppendExecutionProvider_TensorRT(session_options, &o);
    if(onnx_status != NULL)
    {
        const char* msg = ort->GetErrorMessage(onnx_status);
        LOG_ERROR("open openvino error %s", msg);
        ort->ReleaseStatus(onnx_status);
        return ERROR;
    }
    //to do 只是声明内存已经在GPU上了，指针是GPU指针,所以创建的时候需要用GPU malloc 
    // cudaMalloc(&gpu_input, size); cudaMemcpy(gpu_input, cpu_input, size, cudaMemcpyHostToDevice);
    //ORT_ERROR(ort->CreateMemoryInfo("Cuda", OrtDeviceAllocator, 0, OrtMemTypeDefault, &memory_info));
 
    LOG_DEBUG("use tensorRT success");
    return SUCCESS;
}

int onnx_use_gpu(int gpu_type)
{
    device_type = gpu_type;
}

int init_onnx()
{
    if(is_init)
    {
        is_init++;
        return SUCCESS;
    }

    ort = OrtGetApiBase()->GetApi(ORT_API_VERSION);
    if(NULL == ort)
    {
        LOG_ERROR("Failed to init ONNX Runtime engine.");
        return ERROR;
    }

    ORT_ERROR(ort->CreateEnv(ORT_LOGGING_LEVEL_WARNING, "onnx runtime", &env));
    
    is_init = 1;
    return SUCCESS;
}

void deinit_onnx()
{
    if(ort && is_init == 1)
    {
        ort->ReleaseMemoryInfo(memory_info);
        ort->ReleaseEnv(env);
        is_init = 0;
    }
    else
    {
        is_init --;
    }
}

void *onnx_model_alloc(const char *model_data, uint64_t model_size, shape_t *input_shape, 
                        int *in_nodes,shape_t *output_shape, int *out_nodes, void *user_data)
{
    int i, j;
    int thread_num = 1;

    OrtSessionOptions* session_options = NULL;
	OrtStatus *status = NULL;
    OrtTypeInfo* input_info = NULL, *output_info = NULL;
    OrtTensorTypeAndShapeInfo* input_tensor_info, *output_tensor_info;

    ONNXTensorElementDataType onnx_data_type;

    OrtAllocator *allocator = NULL;

    //int64_t* output_shape64 = NULL;
    //int64_t* input_shape64 = NULL;
    size_t num_dims = 0;

    struct onnx_handle_t *onnx_handle = (struct onnx_handle_t *)malloc(sizeof(struct onnx_handle_t));
    if(NULL == onnx_handle)
    {   
        LOG_ERROR("onnx handle malloc size %ld error %s", sizeof(struct onnx_handle_t), strerror(errno));
        return NULL;
    }   
    memset(onnx_handle, 0, sizeof(struct onnx_handle_t));

    // session_options_.SetGraphOptimizationLevel(ORT_ENABLE_ALL);
    //

	ORT_ERROR(ort->CreateSessionOptions(&session_options));

	ORT_ERROR(ort->SetIntraOpNumThreads(session_options, thread_num));
	ORT_ERROR(ort->SetInterOpNumThreads(session_options, thread_num));
	//ORT_ERROR(ort->DisablePerSessionThreads(&session_options));

	ORT_ERROR(ort->SetSessionGraphOptimizationLevel(&session_options, ORT_ENABLE_ALL));

    switch(device_type)
    {
        case USE_CUDA:
            onnx_enable_cuda(session_options);
            //to do 
            //ORT_ERROR(ort->CreateMemoryInfo("Cuda", OrtDeviceAllocator, 0, OrtMemTypeDefault, &memory_info));
            ORT_ERROR(ort->CreateCpuMemoryInfo(OrtDeviceAllocator, OrtMemTypeDefault, &memory_info));
            break;

        case USE_OPENVINO:
            onnx_enable_openvino(session_options);
	        ORT_ERROR(ort->CreateCpuMemoryInfo(OrtArenaAllocator, OrtMemTypeDefault, &memory_info));
            break;

        case USE_TENSOR_RT:
            onnx_enable_tensorRT(session_options);
            //to do 
            //ORT_ERROR(ort->CreateMemoryInfo("Cuda", OrtDeviceAllocator, 0, OrtMemTypeDefault, &memory_info));
            ORT_ERROR(ort->CreateCpuMemoryInfo(OrtDeviceAllocator, OrtMemTypeDefault, &memory_info));
            break;
        case USE_CPU:
	        ORT_ERROR(ort->CreateCpuMemoryInfo(OrtArenaAllocator, OrtMemTypeDefault, &memory_info));
        default:
            break;
    }


    if(model_size > 0)
    {
        ORT_ERROR(ort->CreateSessionFromArray(env, model_data, model_size, session_options, &onnx_handle->session));
    }
    else
    {
        ORT_ERROR(ort->CreateSession(env, model_data, session_options, &onnx_handle->session));
    }

	ORT_ERROR(ort->SessionGetInputCount(onnx_handle->session, &onnx_handle->in_nodes));
	ORT_ERROR(ort->SessionGetOutputCount(onnx_handle->session, &onnx_handle->out_nodes));
	ORT_ERROR(ort->GetAllocatorWithDefaultOptions(&allocator));

    for(i = 0; i < onnx_handle->in_nodes; i++)
    {
        ORT_ERROR(ort->SessionGetInputTypeInfo(onnx_handle->session, i, &input_info));
        ORT_ERROR(ort->CastTypeInfoToTensorInfo(input_info, &input_tensor_info));
        ORT_ERROR(ort->GetDimensionsCount(input_tensor_info, &num_dims));

        ORT_ERROR(ort->GetTensorElementType(input_tensor_info, &onnx_data_type));

        switch(onnx_data_type)
        {
            case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT:
                input_shape[i].data_type = TENSOR_FLOAT32;
                break;
            case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32:
                input_shape[i].data_type = TENSOR_INT32;
                break;
            case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64:
                input_shape[i].data_type = TENSOR_INT64;
                break;
            default:
                input_shape[i].data_type = TENSOR_FLOAT32;
                break;
        }
        
        input_shape[i].num_dims = num_dims;

        ORT_ERROR(ort->SessionGetInputName(onnx_handle->session, i, allocator, &onnx_handle->input_names[i]));

        //input_shape64 = (int64_t*)malloc(num_dims * sizeof(int64_t));
        int64_t input_shape64[num_dims];

        ORT_ERROR(ort->GetDimensions(input_tensor_info, input_shape64, num_dims));
        input_shape[i].num_dims = num_dims;

        for(j = 0; j < num_dims; j++)
        {
            input_shape[i].dims[j] = input_shape64[j];
        }
        LOG_DEBUG("input[%d] name %s data_type %d ", i, onnx_handle->input_names[i], input_shape[i].data_type);

        strncpy(input_shape[i].name, onnx_handle->input_names[i], sizeof(input_shape[i].name));
        print_shape(input_shape[i]);
        ort->ReleaseTypeInfo(input_info);
        //free(input_shape64);
    }

    for(i = 0; i < onnx_handle->out_nodes; i++)
    {
        ORT_ERROR(ort->SessionGetOutputTypeInfo(onnx_handle->session, i, &output_info));
        ORT_ERROR(ort->CastTypeInfoToTensorInfo(output_info, &output_tensor_info));
        ORT_ERROR(ort->GetDimensionsCount(output_tensor_info, &num_dims));

        ORT_ERROR(ort->SessionGetOutputName(onnx_handle->session, i, allocator, &onnx_handle->output_names[i]));

        ORT_ERROR(ort->GetTensorElementType(output_tensor_info, &onnx_data_type));
        switch(onnx_data_type)
        {
            case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT:
                output_shape[i].data_type = TENSOR_FLOAT32;
                break;
            case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32:
                output_shape[i].data_type = TENSOR_INT32;
                break;
            case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64:
                output_shape[i].data_type = TENSOR_INT64;
                break;
            default:
                output_shape[i].data_type = TENSOR_FLOAT32;
                break;
        }

        //output_shape64 = (int64_t*)malloc(num_dims * sizeof(int64_t));
        int64_t output_shape64[num_dims];;

        ORT_ERROR(ort->GetDimensions(output_tensor_info, output_shape64, num_dims));
        output_shape[i].num_dims = num_dims;

        for(j = 0; j < num_dims; j++)
        {
            output_shape[i].dims[j] = output_shape64[j];
        }

        ort->ReleaseTypeInfo(output_info);
        LOG_DEBUG("output[%d] name %s data_type %d ", i, onnx_handle->output_names[i], output_shape[i].data_type);
        strncpy(output_shape[i].name, onnx_handle->output_names[i], sizeof(output_shape[i].name));
        print_shape(output_shape[i]);
        //free(output_shape64);
    }

    *in_nodes = onnx_handle->in_nodes;
    *out_nodes = onnx_handle->out_nodes;

    //onnx_handle->user_data = user_data;

    ort->ReleaseSessionOptions(session_options);
    return onnx_handle;
}



int onnx_data_alloc(void *handle,
         int in_nodes, shape_t *input_shape, matrix_t **input_matrix, int in_alloc_flag,
         int out_nodes, shape_t *output_shape, matrix_t **output_matrix, int out_alloc_flag)
{
    int i, j;
    struct onnx_handle_t *onnx_handle = (struct onnx_handle_t *)handle;

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
        onnx_handle->is_dynamic = 1;
    }

    return SUCCESS;
}

int onnx_data_free(void *handle, matrix_t **input_matrix, matrix_t **output_matrix)
{
    int i;
    struct onnx_handle_t *onnx_handle = (struct onnx_handle_t *)handle;

    if(NULL == onnx_handle)
    {
        return ERROR;
    }

    for(i = 0; i < onnx_handle->in_nodes; i++)
    {
        if(input_matrix[i])
        {
            matrix_free(input_matrix[i]);
        }
        input_matrix[i] = NULL;
    }

    for(i = 0; i < onnx_handle->out_nodes; i++)
    {
        if(output_matrix[i])
            matrix_free(output_matrix[i]);
        output_matrix[i] = NULL;
    }
    return SUCCESS;
}


void onnx_model_free(void *handle)
{
    int i;
    struct onnx_handle_t *onnx_handle = (struct onnx_handle_t *)handle;
    if(NULL == onnx_handle)
    {
        return;
    }

    for(i = 0; i < onnx_handle->in_nodes; i++)
    {
        free(onnx_handle->input_names[i]);
    }

    for(i = 0; i < onnx_handle->out_nodes; i++)
    {
        free(onnx_handle->output_names[i]);
    }

    ort->ReleaseSession(onnx_handle->session);
    free(onnx_handle);
}

int onnx_inference(void *handle, void *input_data, void *output_data, int is_debug)
{
    int i, j;
    void *onnx_output_data = NULL;
    uint64_t start_time, end_time;
    struct onnx_handle_t *onnx_handle = (struct onnx_handle_t *)handle;
    OrtTypeInfo* input_info = NULL, *output_info = NULL;
    OrtTensorTypeAndShapeInfo* input_tensor_info = NULL, *output_tensor_info = NULL;
    ONNXTensorElementDataType data_type;

    matrix_t **input_matrix = (matrix_t **)input_data;
    matrix_t **output_matrix = (matrix_t **)output_data;

    if(is_debug)
        start_time = get_ustime();

    for(i = 0; i < onnx_handle->in_nodes; i++)
    {
        int64_t input_shape64[input_matrix[i]->shape.num_dims];;
        for(j = 0; j < input_matrix[i]->shape.num_dims; j++)
        {
            input_shape64[j] = input_matrix[i]->shape.dims[j];
        }

        switch(input_matrix[i]->shape.data_type)
        {
            case TENSOR_INT32:
        	    ORT_ERROR(ort->CreateTensorWithDataAsOrtValue(memory_info, 
            	    input_matrix[i]->data, input_matrix[i]->shape.size * sizeof(int32_t),
            	    input_shape64, input_matrix[i]->shape.num_dims, ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32,
            	    &onnx_handle->input_tensor[i]));

                break;
            case TENSOR_INT64:
        	    ORT_ERROR(ort->CreateTensorWithDataAsOrtValue(memory_info, 
            	    input_matrix[i]->data, input_matrix[i]->shape.size * sizeof(int64_t),
            	    input_shape64, input_matrix[i]->shape.num_dims, ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64,
            	    &onnx_handle->input_tensor[i]));

                break;
            case TENSOR_FLOAT32:
        	    ORT_ERROR(ort->CreateTensorWithDataAsOrtValue(memory_info, 
            	    input_matrix[i]->data, input_matrix[i]->shape.size * sizeof(float),
            	    input_shape64, input_matrix[i]->shape.num_dims, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,
            	    &onnx_handle->input_tensor[i]));
                break;
            default:
                break;
        }
    }

    if(onnx_handle->is_dynamic)   //output dynamic memeroy
    {
        LOG_DEBUG("to do dynamic mem ");
        //to do memcpy check
        ORT_ERROR(ort->Run(onnx_handle->session, NULL, onnx_handle->input_names, 
                    (const OrtValue* const*)&onnx_handle->input_tensor, onnx_handle->in_nodes, 
					onnx_handle->output_names, onnx_handle->out_nodes, &onnx_handle->output_tensor)); 
        for(i = 0; i < onnx_handle->out_nodes; i++)
        {
	        int64_t output_shape64[output_matrix[i]->shape.num_dims];;
            ORT_ERROR(ort->GetTensorMutableData(onnx_handle->output_tensor[i], (void**)&onnx_output_data));
            ORT_ERROR(ort->GetTensorTypeAndShape(onnx_handle->output_tensor[i], &output_tensor_info));
            ORT_ERROR(ort->GetDimensions(output_tensor_info, output_shape64, output_matrix[i]->shape.num_dims));
            for(j = 0; j < output_matrix[i]->shape.num_dims; j++)
        	{
                output_matrix[i]->shape.dims[j] = output_shape64[j];
            }
            matrix_resize_shape(output_matrix[i], output_matrix[i]->shape);
            memcpy(output_matrix[i]->data, onnx_output_data, output_matrix[i]->shape.tensor_size);
            ort->ReleaseTensorTypeAndShapeInfo(output_tensor_info);
        }
    }
    else
    {
	    for(i = 0; i < onnx_handle->out_nodes; i++)
	    {
	        int64_t output_shape64[output_matrix[i]->shape.num_dims];;
	        for(j = 0; j < output_matrix[i]->shape.num_dims; j++)
	        {
	            output_shape64[j] = output_matrix[i]->shape.dims[j];
	        }
	
	        switch(output_matrix[i]->shape.data_type)
	        {
	            case TENSOR_INT32:
	        	    ORT_ERROR(ort->CreateTensorWithDataAsOrtValue(memory_info, 
	            	    output_matrix[i]->data, output_matrix[i]->shape.size * sizeof(int32_t),
	            	    output_shape64, output_matrix[i]->shape.num_dims, ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32,
	            	    &onnx_handle->output_tensor[i]));
	                break;
	            case TENSOR_INT64:
	        	    ORT_ERROR(ort->CreateTensorWithDataAsOrtValue(memory_info, 
	            	    output_matrix[i]->data, output_matrix[i]->shape.size * sizeof(int64_t),
	            	    output_shape64, output_matrix[i]->shape.num_dims, ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64,
	            	    &onnx_handle->output_tensor[i]));
	                break;
	            case TENSOR_FLOAT32:
	        	    ORT_ERROR(ort->CreateTensorWithDataAsOrtValue(memory_info, 
	            	    output_matrix[i]->data, output_matrix[i]->shape.size * sizeof(float),
	            	    output_shape64, output_matrix[i]->shape.num_dims, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,
	            	    &onnx_handle->output_tensor[i]));
	                break;
	            default:
	                break;
	        }
	    }
        ORT_ERROR(ort->Run(onnx_handle->session, NULL, onnx_handle->input_names, 
                (const OrtValue* const*)&onnx_handle->input_tensor, onnx_handle->in_nodes, 
				onnx_handle->output_names, onnx_handle->out_nodes, &onnx_handle->output_tensor));
    }
        
  	if(is_debug)
    {
        end_time = get_ustime();
        LOG_DEBUG("once run device %s use %lld  ms ", device_name[device_type],  (end_time - start_time)/ 1000);
    }

    for(i = 0; i < onnx_handle->in_nodes; i++)
    {
        ort->ReleaseValue(onnx_handle->input_tensor[i]);
        onnx_handle->input_tensor[i] = NULL;
    }

    for(i = 0; i < onnx_handle->out_nodes; i++)
    {
        ort->ReleaseValue(onnx_handle->output_tensor[i]);
        onnx_handle->output_tensor[i] = NULL;
    }
    return SUCCESS;
}


#define ONNX_MODEL_TEST 0

#if ONNX_MODEL_TEST
int main(int argc, char *argv[])
{
    //input cache output[1]->cache
    const char *path = "weights/wekws.onnx";
    //const char *path = argv[1];
    void *onnx_handle = NULL;
    //const char *path = "";

    shape_t input_shape[4], output_shape[4];
    int in_nodes, out_nodes;
    matrix_t *input_matrix[4];
    matrix_t *output_matrix[4];

    init_onnx();
    onnx_handle = onnx_model_alloc(path, input_shape, &in_nodes, output_shape, &out_nodes, NULL);


    input_shape[0].dims[1] = 80;
    output_shape[0].dims[1] = 80;
    output_shape[1].dims[2] = 105;

    onnx_alloc_data_mem(onnx_handle, in_nodes, input_shape, input_matrix, out_nodes, output_shape, output_matrix);

    print_shape(output_matrix[0]->shape, "output");
    print_shape(output_matrix[1]->shape, "output");

    onnx_inference(onnx_handle, NULL, output_matrix);
    print_matrix(output_matrix[0], "out");

    memcpy(input_matrix[1]->data, output_matrix[1]->data, sizeof(float) * output_matrix[1]->shape.size);
    onnx_inference(onnx_handle, NULL, output_matrix);

    print_matrix(output_matrix[0], "out");

} 
#endif //ONNX_MODEL_TEST

#endif
