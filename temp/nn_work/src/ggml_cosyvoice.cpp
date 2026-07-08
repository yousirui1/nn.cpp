#include "base.h"

#include "cosyvoice.h"

#include "audio_file.h"
#include "cosyvoice_frontend.h"
#include "gguf_loader.h"
#include "ggml_module.h"
#include "ggml_model.h"
#include "ggml_cosyvoice.h"
#include <filesystem>

struct cosyvoice_frontend_deleter { void operator()(cosyvoice_frontend_context_t ctx) const noexcept { cosyvoice_frontend_free(ctx); } };

struct cosyvoice_prompt_speech_deleter { void operator()(cosyvoice_prompt_speech_t prompt_speech) const noexcept { cosyvoice_prompt_speech_free(prompt_speech); } };

struct cosyvoice_context_deleter { void operator()(cosyvoice_context_t ctx) const noexcept { cosyvoice_free(ctx); } };

struct cosyvoice_prompt_deleter { void operator()(cosyvoice_prompt_t prompt) const noexcept { cosyvoice_prompt_free(prompt); } };

struct audio_buffer_deleter { void operator()(float* data) const noexcept { free_audio_data(data); } };


using audio_buffer_handle = std::unique_ptr<float, audio_buffer_deleter>;
using cosyvoice_frontend_handle = std::unique_ptr<cosyvoice_frontend_context, cosyvoice_frontend_deleter>;
#if 0
using cosyvoice_prompt_speech_handle = std::unique_ptr<cosyvoice_prompt_speech, cosyvoice_prompt_speech_deleter>;
using cosyvoice_context_handle = std::unique_ptr<cosyvoice_context, cosyvoice_context_deleter>;
using cosyvoice_prompt_handle = std::unique_ptr<cosyvoice_prompt, cosyvoice_prompt_deleter>;
#endif


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

int load_cosyvoice_model(struct ggml_handle_t *ggml_handle, const char *model_path,
                        shape_t *input_shape, int *in_nodes,shape_t *output_shape, int *out_nodes)
{
    float* prompt_audio_data = nullptr;
    uint32_t prompt_audio_length = 0;
    uint32_t prompt_audio_sample_rate = 0;
    int max_llm_len = 2048;
    bool text_normalization_enabled = true;


    //cosyvoice_prompt_speech_handle prompt_speech;
    std::filesystem::path p(model_path);
    std::string prompt_model = p.parent_path().string() + "/prompt.gguf";
    cosyvoice_init_backend();

    cosyvoice_context_params_t params;
    cosyvoice_init_default_context_params(&params);
    params.inference_buffer_policy = COSYVOICE_INFERENCE_BUFFER_POLICY_SHARED;
    params.n_max_seq = max_llm_len;

    struct cosyvoice_model_t *model = (struct cosyvoice_model_t *)malloc(sizeof(struct cosyvoice_model_t));
    if(!model)
    {
        LOG_ERROR("cosyvoice_model_t malloc sizeof %ld error %s", sizeof(struct cosyvoice_model_t), strerror(errno));
        return ERROR;
    }
    memset(model, 0, sizeof(struct cosyvoice_model_t));

    model->prompt_speech = cosyvoice_prompt_speech_load_from_file(prompt_model.c_str());
    if (!model->prompt_speech)
    {
        LOG_ERROR("failed to load prompt_speech file: prompt.gguf");
        if (errno != 0)
            LOG_ERROR("Reason: %s", strerror(errno));
        return ERROR;
    }


    model->ctx = cosyvoice_load_from_file_with_params(model_path, &params);

    if (!model->ctx)
    {
        LOG_ERROR("failed to load model file \"%s\".", model_path);
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

    ggml_handle->model = (void *)model;
    *in_nodes = 1; 
    set_shape(&input_shape[0], TENSOR_FLOAT32, 1, -1); 

    *out_nodes = 1; 
    set_shape(&output_shape[0], TENSOR_FLOAT32, 1, 16000 * 60); //60s

    return SUCCESS;
}


int cosyvoice_inference(struct ggml_handle_t *ggml_handle, matrix_t **input_matrix, matrix_t **output_matrix, int n_threads, void *param)
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

void unload_cosyvoice_model()
{


}

