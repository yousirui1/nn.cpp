#ifndef __MODEL_LIST_H__
#define __MODEL_LIST_H__

#include "ggml_silero.h"
#include "ggml_sensevoice.h"
#include "ggml_cosyvoice.h"

struct ggml_model_t
{
    char name[128];

    int (*load_model)(struct ggml_handle_t *ggml_handle, const char *model_data, int model_size);

    int (*inference)(struct ggml_handle_t *ggml_handle, matrix_t **input_matrix, matrix_t **output_matrix);
    void (*unload_model)(struct ggml_handle_t *ggml_handle);
};

static struct ggml_model_t ggml_models[] =
{
    //name                  load                    inference                   unload
    //{"SenseVoiceSmall",   load_silero_model,       silero_inference ,     unload_silero_model},
    {"SenseVoiceSmall", load_sensevoice_model,  sensevoice_inference,   unload_sensevoice_model},

    //{"cosyvoice3-2512", load_cosyvoice_model,  cosyvoice_inference ,    unload_cosyvoice_model},
    //{"FsmnVADStreaming",    load_fsmn_model,         fsmn_inference,            unload_fsmn_model},
};


#endif //  __MODEL_LIST_H__
