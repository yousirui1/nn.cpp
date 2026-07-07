#ifndef COSYVOICE_LOWLEVEL_H
#define COSYVOICE_LOWLEVEL_H



#include <ggml.h>
#include <ggml-alloc.h>

#ifdef __cplusplus
extern "C" {
#endif // __cplusplus

typedef struct cosyvoice_tokenization_result* cosyvoice_tokenization_result_t;
typedef struct cosyvoice_tokenizer_context*   cosyvoice_tokenizer_context_t;  


typedef enum cosyvoice_inference_mode
{
	COSYVOICE_INFERENCE_MODE_NULL = -1,     
	COSYVOICE_INFERENCE_MODE_ZERO_SHOT,      
	COSYVOICE_INFERENCE_MODE_INSTRUCT,      
	COSYVOICE_INFERENCE_MODE_CROSS_LINGUAL, 
	COSYVOICE_INFERENCE_MODE_COUNT          
} cosyvoice_inference_mode_t;


typedef enum cosyvoice_noise_callback_stage
{
	COSYVOICE_NOISE_CALLBACK_STAGE_BEFORE_FLOW, 
	COSYVOICE_NOISE_CALLBACK_STAGE_AFTER_FLOW, 
	COSYVOICE_NOISE_CALLBACK_STAGE_BEFORE_HIFT, 
	COSYVOICE_NOISE_CALLBACK_STAGE_AFTER_HIFT  
} cosyvoice_noise_callback_stage_t;


typedef float* (*cosyvoice_noise_callback_t)(
	cosyvoice_noise_callback_stage_t stage,
	uint32_t                         length,
	float*                           noise,
	void*                            ctx
);


#define COSYVOICE_CONTEXT_PARAMS_VERSION     (0ul)
#define COSYVOICE_CONTEXT_PARAMS_V2_VERSION  (1ul)


 void cosyvoice_log_callback_default(enum ggml_log_level level, const char* text, void* user_data);


 cosyvoice_context_t cosyvoice_load_from_file_ext(
    const char*                       filename,
    const cosyvoice_context_params_t* params,
    ggml_backend_t                    backend,
    uint32_t                          n_threads,
    uint32_t                          params_version
);

#ifdef __cplusplus
}

namespace cosyvoice::internal_utils
{
template<typename T, typename U>
struct _is_same { static constexpr bool value = false; };
template<typename T>
struct _is_same<T, T> { static constexpr bool value = true; };
}


template<typename params_t>
inline cosyvoice_context_t cosyvoice_load_from_file_ext(
    const char*                       filename,
    const params_t*                   params,
    ggml_backend_t                    backend,
    uint32_t                          n_threads
)
{
    using namespace cosyvoice::internal_utils;

    if constexpr (_is_same<params_t, cosyvoice_context_params_t>::value)
        return cosyvoice_load_from_file_ext(filename, params, backend, n_threads, COSYVOICE_CONTEXT_PARAMS_VERSION);
    else if constexpr (_is_same<params_t, cosyvoice_context_params_v2_t>::value)
        return cosyvoice_load_from_file_ext(filename, &params->base_params, backend, n_threads, COSYVOICE_CONTEXT_PARAMS_V2_VERSION);
    else
    {
        static_assert(_is_same<params_t, cosyvoice_context_params_v2_cpp>::value, "Unsupported context parameter type");
        return cosyvoice_load_from_file_ext(filename, params, backend, n_threads, COSYVOICE_CONTEXT_PARAMS_V2_VERSION);
    }
}

extern "C" {
#endif

 enum ggml_status cosyvoice_get_last_status(cosyvoice_context_t ctx);


 const ggml_tensor* cosyvoice_get_word_token_embed_weight(cosyvoice_context_t ctx);


 const ggml_tensor* cosyvoice_get_speech_token_embed_weight(cosyvoice_context_t ctx);

 bool cosyvoice_llm_prefill(
    cosyvoice_context_t ctx,
    enum ggml_type      type,
    const void*         data,
    uint32_t            n_tokens
);


 bool cosyvoice_llm_decode(
    cosyvoice_context_t ctx,
    enum ggml_type      type,
    const void*         data
);


 void cosyvoice_llm_prepare_probs(cosyvoice_context_t ctx, bool allow_stop_tokens);


 uint32_t cosyvoice_llm_get_kv_cache_len(cosyvoice_context_t ctx);

bool cosyvoice_llm_set_kv_cache_len(cosyvoice_context_t ctx, uint32_t len);

int  cosyvoice_llm_sample_token(cosyvoice_context_t ctx);

bool cosyvoice_llm_is_stop_token(cosyvoice_context_t ctx, int token_id);

void cosyvoice_llm_accept_token(cosyvoice_context_t ctx, int token_id);

void cosyvoice_llm_clear_accepted_tokens(cosyvoice_context_t ctx);

uint32_t   cosyvoice_llm_get_n_accepted_tokens(cosyvoice_context_t ctx);

const int* cosyvoice_llm_get_accepted_tokens(cosyvoice_context_t ctx);

bool cosyvoice_llm_job(
	cosyvoice_context_t ctx,
	const int*          text,
	uint32_t            text_len,
	cosyvoice_prompt_t  prompt
);

bool cosyvoice_token2wav(
	cosyvoice_context_t            ctx,
	const int*                     token_ids,
	uint32_t                       n_tokens,
	float                          speed,
	cosyvoice_prompt_t             prompt,
	cosyvoice_generated_speech_ptr generated_speech
);

bool cosyvoice_tts(
	cosyvoice_context_t            ctx,
	const int*                     text,
	uint32_t                       text_len,
	float                          speed,
	cosyvoice_prompt_t             prompt,
	cosyvoice_generated_speech_ptr result
);

cosyvoice_tokenizer_context_t   cosyvoice_get_tokenizer(cosyvoice_context_t ctx);

cosyvoice_tokenizer_context_t   cosyvoice_tokenizer_load_from_file(const char* filename);

void cosyvoice_tokenizer_free(cosyvoice_tokenizer_context_t ctx);

cosyvoice_tokenization_result_t cosyvoice_tokenization_result_create();

void cosyvoice_tokenization_result_free(cosyvoice_tokenization_result_t result);

int* cosyvoice_tokenization_result_get_tokens(cosyvoice_tokenization_result_t result);

uint32_t cosyvoice_tokenization_result_get_n_tokens(cosyvoice_tokenization_result_t result);

uint32_t cosyvoice_tokenize(
	cosyvoice_tokenizer_context_t   ctx,
	const char*                     text,
	cosyvoice_tokenization_result_t result,
	bool                            parse_special
);

uint32_t cosyvoice_tokenize_ext(
	cosyvoice_tokenizer_context_t   ctx,
	const char*                     text,
	uint32_t                        text_len,
	cosyvoice_tokenization_result_t result,
	bool                            parse_special
);

void cosyvoice_set_noise_callback(cosyvoice_context_t ctx, cosyvoice_noise_callback_t callback, void* callback_ctx);

void cosyvoice_get_noise_callback(cosyvoice_context_t ctx, cosyvoice_noise_callback_t* callback, void** callback_ctx);

uint32_t cosyvoice_get_hift_rand_ini_len(cosyvoice_context_t ctx);

void cosyvoice_set_hift_rand_ini(cosyvoice_context_t ctx, const float* data);

uint32_t cosyvoice_prompt_speech_get_crc32(cosyvoice_prompt_speech_t prompt_speech);

uint32_t cosyvoice_prompt_get_crc32(cosyvoice_prompt_t prompt);

const char* cosyvoice_get_instruction_prefix(cosyvoice_context_t ctx);

cosyvoice_prompt_t cosyvoice_prompt_set(
	cosyvoice_context_t        ctx,
	cosyvoice_prompt_t         prompt,
	cosyvoice_inference_mode_t mode,
	const char*                instruction,
#ifdef __cplusplus
	uint32_t                   instruction_length = 0xFFFFFFFFU,
	bool                       inplace = true
#else
	uint32_t                   instruction_length,
	bool                       inplace
#endif
);

cosyvoice_prompt_t cosyvoice_prompt_set_ext(
	cosyvoice_context_t        ctx,
	cosyvoice_prompt_t         prompt,
	cosyvoice_inference_mode_t mode,
	const int*                 instruction,
	uint32_t                   instruction_length,
	bool                       inplace
);

#ifdef __cplusplus
}
#endif // __cplusplus

#endif // COSYVOICE_LOWLEVEL_H
