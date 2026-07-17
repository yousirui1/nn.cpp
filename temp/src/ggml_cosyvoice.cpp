#include "base.h"

#include "cosyvoice.h"
#include "matrix.h"
#include "audio_file.h"
//#include "cosyvoice_frontend.h"
#include "gguf_loader.h"
#include "ggml_module.h"
#include "ggml_model.h"
#include "ggml_cosyvoice.h"
#include <filesystem>
#include "ggml_cosyvoice.h"

struct cosyvoice_model_t
{
    cosyvoice_prompt_speech_t prompt_speech;
    cosyvoice_context_t ctx;
    cosyvoice_prompt_t prompt;
    cosyvoice_tts_context_t tts_ctx;
};


void get_cosyvoice_default_params(struct cosyvoice_params_t *params)
{
    //to do 
};


int load_cosyvoice_model(struct ggml_handle_t *ggml_handle, const char *model_data, int model_size)
{
    float* prompt_audio_data = nullptr;
    uint32_t prompt_audio_length = 0;
    uint32_t prompt_audio_sample_rate = 0;
    int max_llm_len = 2048;
    bool text_normalization_enabled = true;

    //cosyvoice_prompt_speech_handle prompt_speech;
    cosyvoice_init_backend();

    struct cosyvoice_model_t *model = (struct cosyvoice_model_t *)malloc(sizeof(struct cosyvoice_model_t));
    if(!model)
    {
        LOG_ERROR("cosyvoice_model_t malloc sizeof %ld error %s", sizeof(struct cosyvoice_model_t), strerror(errno));
        return ERROR;
    }
    memset(model, 0, sizeof(struct cosyvoice_model_t));

    if(model_size)
    {
        model->ctx= cosyvoice_duplicate_context(((struct cosyvoice_model_t *)((struct ggml_handle_t *)model_data)->model)->ctx);

        model->prompt = ((struct cosyvoice_model_t *)((struct ggml_handle_t *)model_data)->model)->prompt;

        model->tts_ctx = cosyvoice_tts_context_new(model->ctx, model->prompt);
        if (!model->tts_ctx)
        {
            LOG_ERROR("Error: failed to create TTS context.");
            return ERROR;
        }
        ggml_handle->is_dup = 1;
    }
    else
    {
        cosyvoice_context_params_t params;
        cosyvoice_init_default_context_params(&params);
        params.inference_buffer_policy = COSYVOICE_INFERENCE_BUFFER_POLICY_SHARED;
        params.n_max_seq = max_llm_len;

        std::filesystem::path p(model_data);
        std::string prompt_model = p.parent_path().string() + "/prompt.gguf";

        model->prompt_speech = cosyvoice_prompt_speech_load_from_file(prompt_model.c_str());
        if (!model->prompt_speech)
        {
            LOG_ERROR("failed to load prompt_speech file: prompt.gguf");
            if (errno != 0)
                LOG_ERROR("Reason: %s", strerror(errno));
            return ERROR;
        }

        model->ctx = cosyvoice_load_from_file_with_params(model_data, &params);

        if (!model->ctx)
        {
            LOG_ERROR("failed to load model file \"%s\".", model_data);
            return ERROR;
        }

        model->prompt = cosyvoice_prompt_init_from_prompt_speech(model->ctx, model->prompt_speech);
        if (!model->prompt)
        {
            LOG_ERROR("failed to initialize prompt from prompt_speech.");
            return ERROR;
        }

        model->tts_ctx = cosyvoice_tts_context_new(model->ctx, model->prompt);
        if (!model->tts_ctx)
        {
            LOG_ERROR("Error: failed to create TTS context.");
            return ERROR;
        }

#ifndef COSYVOICE_NO_ICU
        cosyvoice_tts_context_set_text_normalization_enabled(model->tts_ctx, text_normalization_enabled);
#endif
        ggml_handle->is_dup = 0;
    }

    ggml_handle->model = (void *)model;
    ggml_handle->in_nodes = 1;
    set_shape(&ggml_handle->input_shape[0], TENSOR_FLOAT32, 1, -1);

    ggml_handle->out_nodes = 1;
    set_shape(&ggml_handle->output_shape[0], TENSOR_FLOAT32, 1, 16000 * 60); //60s

    return SUCCESS;
}

int cosyvoice_inference(struct ggml_handle_t *ggml_handle, matrix_t **input_matrix, matrix_t **output_matrix, void *param)
{
    struct cosyvoice_model_t *model = (struct cosyvoice_model_t *)ggml_handle->model;

    float speed = 1.0f;
    std::string instruction = "";
    std::string mode = "auto";
    //std::string output = "output2.wav";

    cosyvoice_generated_speech result = {};
    bool ok = false;
    if (mode == "cross-lingual")
        ok = cosyvoice_tts_cross_lingual(model->tts_ctx, (char *)&input_matrix[0]->data_i8[0], speed, &result);
    else if (mode == "zero-shot")
        ok = cosyvoice_tts_zero_shot(model->tts_ctx, (char *)&input_matrix[0]->data_i8[0], speed, &result);
    else
        ok = cosyvoice_tts_instruct(model->tts_ctx, (char *)&input_matrix[0]->data_i8[0], instruction.c_str(), speed, &result);


    if(output_matrix[0]->shape.size >= result.length)
    {
        memcpy(output_matrix[0]->data_fp, result.data, result.length * sizeof(float));
        output_matrix[0]->shape.dims[0] = result.length;
    }

    return SUCCESS;
}

void unload_cosyvoice_model(struct ggml_handle_t *ggml_handle)
{
    if(!ggml_handle->model)
    {
        return;
    }
    struct cosyvoice_model_t *model = (struct cosyvoice_model_t *)ggml_handle->model;

    if(model->tts_ctx)
    {
        cosyvoice_tts_context_free(model->tts_ctx);
        model->tts_ctx = NULL;
    }

    if(model->ctx)
    {   
        cosyvoice_free(model->ctx);
        model->ctx = NULL;
    }

    if(!ggml_handle->is_dup)
    {
        if(model->prompt)
        {   
            cosyvoice_prompt_free(model->prompt);
            model->prompt = NULL;
        }   
        if(model->prompt_speech)
        {   
            cosyvoice_prompt_speech_free(model->prompt_speech);
            model->prompt_speech = NULL;
        }   
    }
    free(model);
    ggml_handle->model = NULL;
}


