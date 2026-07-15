#include "base.h"
#include "tools.h"
#include "ggml_model.h"

int main(int argc, char *argv[])
{
    int in_nodes, out_nodes;
    shape_t input_shape[MAX_NUM_LAYER];
    shape_t output_shape[MAX_NUM_LAYER];

    matrix_t *input_matrix[MAX_NUM_LAYER];
    matrix_t *output_matrix[MAX_NUM_LAYER];

    char *model_path = "/home/ysr/project/models/gguf/Fun-CosyVoice3-0.5B-2512-GGUF/CosyVoice3-2512_F32.gguf";
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

    struct ggml_handle_t *handle = ggml_model_alloc(model_path, 0, input_shape, &in_nodes, output_shape, &out_nodes, 1, NULL);

    struct ggml_handle_t *handle1 = ggml_model_alloc((char *)(handle), 1, input_shape, &in_nodes, output_shape, &out_nodes, 1, NULL);

    ggml_data_alloc(handle1, in_nodes, input_shape, input_matrix, 0,
                    out_nodes, output_shape, output_matrix, 1);

    char *text = "感谢观赏";

    //memcpy(input_matrix[0]->data_fp, wav_buf, wav_count * sizeof(float));
    set_matrix_data(input_matrix[0], text);
    input_matrix[0]->shape.size = strlen(text);

    ggml_inference(handle1, input_matrix, output_matrix, NULL, 1);

    float *output_data;
    int output_size;

    output_data = (float *)&output_matrix[0]->data_fp[0];
    output_size = output_matrix[0]->shape.dims[0];

    write_wav_file("output.wav", output_data, output_size, sizeof(float), 24000);

    //printf("asr result size %d msg %s\n",  output_size, output_data);

#if 0
    //while(1)
    {
    offset = 0;
    do{
        memcpy(input_matrix[1]->data_fp, wav_buf + offset, 512 * sizeof(float));
        ggml_inference(handle1, input_matrix, output_matrix, NULL, 1); 
        offset += seg_sample;
        //usleep(seg_time);
    }while(wav_count > seg_sample + offset);
    }
#endif

    free(wav_buf);

    ggml_data_free(handle, input_matrix, output_matrix);
    ggml_model_free(handle);
    deinit_ggml();
    return SUCCESS;

}


