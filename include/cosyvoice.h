#ifndef COSYVOICE_H
#define COSYVOICE_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#else
	#include <stdbool.h>
#endif



typedef enum cosyvoice_llm_kv_cache_type
{
	COSYVOICE_LLM_KV_CACHE_TYPE_F32,  
	COSYVOICE_LLM_KV_CACHE_TYPE_F16,  
	COSYVOICE_LLM_KV_CACHE_TYPE_Q8_0,  
	COSYVOICE_LLM_KV_CACHE_TYPE_Q5_1,  
	COSYVOICE_LLM_KV_CACHE_TYPE_Q5_0,  
	COSYVOICE_LLM_KV_CACHE_TYPE_Q4_1,  
	COSYVOICE_LLM_KV_CACHE_TYPE_Q4_0,  
	COSYVOICE_LLM_KV_CACHE_TYPE_COUNT  
} cosyvoice_llm_kv_cache_type_t;

typedef enum cosyvoice_inference_buffer_policy
{
	COSYVOICE_INFERENCE_BUFFER_POLICY_SHARED,    
	COSYVOICE_INFERENCE_BUFFER_POLICY_BALANCED,  
	COSYVOICE_INFERENCE_BUFFER_POLICY_DEDICATED,
	COSYVOICE_INFERENCE_BUFFER_POLICY_COUNT      
} cosyvoice_inference_buffer_policy_t;

typedef enum cosyvoice_builtin_sampler_rng_policy
{
	COSYVOICE_BUILTIN_SAMPLER_RNG_POLICY_RESET_PER_SESSION,       
	COSYVOICE_BUILTIN_SAMPLER_RNG_POLICY_CONTINUE_ACROSS_SESSIONS, 
	COSYVOICE_BUILTIN_SAMPLER_RNG_POLICY_COUNT                     
} cosyvoice_builtin_sampler_rng_policy_t;

typedef struct cosyvoice_sampling_params
{
	int   top_k;    
	float top_p;   
	int   win_size; 
	float tau_r;    
} cosyvoice_sampling_params_t;

typedef struct cosyvoice_generation_config
{
	float                       temperature;          
	cosyvoice_sampling_params_t sampling;             
	float                       min_token_text_ratio; 
	float                       max_token_text_ratio;
} cosyvoice_generation_config_t;


typedef struct cosyvoice_llm_token_prob
{
	int   token_id; 
	float prob;     
} cosyvoice_llm_token_prob_t;


typedef struct cosyvoice_generated_speech
{
	float*   data;   
	uint32_t length; 
} *cosyvoice_generated_speech_ptr;

typedef struct cosyvoice_context*       cosyvoice_context_t;            
typedef struct cosyvoice_prompt_speech* cosyvoice_prompt_speech_t;     
typedef struct cosyvoice_prompt*        cosyvoice_prompt_t;            
typedef struct cosyvoice_tts_context*   cosyvoice_tts_context_t;      

typedef int (*cosyvoice_sampler_t)(
	cosyvoice_llm_token_prob_t*        nucleus_probs,
	int                                k,
	float*                             probs,
	uint32_t                           size,
	const cosyvoice_sampling_params_t* sampling_params,
	int*                               accepted_tokens,
	uint32_t                           n_accepted_tokens,
	void*                              sampler_ctx
);


typedef int (*cosyvoice_sampler_ext_t)(
    cosyvoice_llm_token_prob_t*        nucleus_probs,
    int                                k,
    float*                             probs,
    uint32_t                           size,
    const cosyvoice_sampling_params_t* sampling_params,
    int*                               accepted_tokens,
    uint32_t                           n_accepted_tokens,
    void*                              sampler_ctx,
    uint32_t                           worker_no
);


#define COSYVOICE_MAKE_SEPARATE_KV_CACHE(k_type, v_type, fallback_type) \
    ((cosyvoice_llm_kv_cache_type_t)((k_type) | ((v_type) << 5) | ((fallback_type) << 10) | (1U << 31)))


#define COSYVOICE_IS_SEPARATE_KV_CACHE(t)  (((t) & (1U << 31)) != 0)


#define COSYVOICE_K_CACHE_TYPE(t)          ((cosyvoice_llm_kv_cache_type_t)((t) & 0x1F))


#define COSYVOICE_V_CACHE_TYPE(t)          ((cosyvoice_llm_kv_cache_type_t)(((t) >> 5) & 0x1F))


#define COSYVOICE_KV_CACHE_FALLBACK(t)     ((cosyvoice_llm_kv_cache_type_t)(((t) >> 10) & 0x1F))


typedef struct cosyvoice_context_params
{
    bool llm_use_flash_attn; 
    bool flow_use_flash_attn;

    union
    {
        struct
        {
#if defined(__BYTE_ORDER__) && (__BYTE_ORDER__ == __ORDER_BIG_ENDIAN__) || defined(_BYTE_ORDER) && (_BYTE_ORDER == _BIG_ENDIAN) || defined(__BIG_ENDIAN__) && !defined(__LITTLE_ENDIAN__) || defined(__ARMEB__) || defined(__MIPSEB__) || defined(__sparc__)
            cosyvoice_llm_kv_cache_type_t  llm_kv_cache_separate_buffers : 1; 
            cosyvoice_llm_kv_cache_type_t : 16;                               
            cosyvoice_llm_kv_cache_type_t  llm_kv_cache_fallback : 5;         
            cosyvoice_llm_kv_cache_type_t  llm_v_cache_type : 5;              
            cosyvoice_llm_kv_cache_type_t  llm_k_cache_type : 5;             
#else
            cosyvoice_llm_kv_cache_type_t  llm_k_cache_type : 5;             
            cosyvoice_llm_kv_cache_type_t  llm_v_cache_type : 5;             
            cosyvoice_llm_kv_cache_type_t  llm_kv_cache_fallback : 5;        
            cosyvoice_llm_kv_cache_type_t : 16;                              
            cosyvoice_llm_kv_cache_type_t  llm_kv_cache_separate_buffers : 1;
#endif
        };
        cosyvoice_llm_kv_cache_type_t      llm_kv_cache_type;                
    };
    bool                                llm_allow_kv_cache_fallback; 
    cosyvoice_inference_buffer_policy_t inference_buffer_policy;     

	uint32_t n_batch;   
	uint32_t n_max_seq; 
	uint32_t seed;      
	cosyvoice_builtin_sampler_rng_policy_t builtin_sampler_rng_policy;

    // Sampling overrides
    union
    {
        cosyvoice_sampler_t sampler;
        cosyvoice_sampler_ext_t sampler_ext;
    };
    void* sampler_ctx;
} cosyvoice_context_params_t;

typedef struct cosyvoice_context_params_v2
{
    cosyvoice_context_params_t base_params;
    uint32_t n_workers;                     
} cosyvoice_context_params_v2_t;

#ifdef __cplusplus
struct cosyvoice_context_params_v2_cpp : cosyvoice_context_params_t
{
    uint32_t n_workers;
};
#endif


 uint32_t cosyvoice_generate_random_seed();


 void cosyvoice_init_backend();


 void cosyvoice_init_backend_from_path(const char* dir_path);


 void     cosyvoice_init_default_context_params(cosyvoice_context_params_t* params);


 cosyvoice_context_t cosyvoice_load_from_file(const char* filename);

 cosyvoice_context_t cosyvoice_load_from_file_with_params(
    const char*                       filename,
    const cosyvoice_context_params_t* params
);


 cosyvoice_context_t cosyvoice_load_from_file_with_params_v2(
    const char*                          filename,
    const cosyvoice_context_params_v2_t* params
);


 cosyvoice_context_t cosyvoice_duplicate_context(cosyvoice_context_t ctx);


 void      cosyvoice_free(cosyvoice_context_t ctx);


 void                cosyvoice_get_context_params(
    cosyvoice_context_t         ctx,
    cosyvoice_context_params_t* params
);


 uint32_t			  cosyvoice_get_n_workers(cosyvoice_context_t ctx);


 uint32_t            cosyvoice_get_worker_no(cosyvoice_context_t ctx);


 bool                cosyvoice_set_worker_no(
    cosyvoice_context_t ctx,
    uint32_t worker_no
);


 const char*         cosyvoice_get_architecture(cosyvoice_context_t ctx);


 bool                cosyvoice_is_backend_uma(cosyvoice_context_t ctx);


 void     cosyvoice_get_default_generation_config(
    cosyvoice_context_t            ctx,
    cosyvoice_generation_config_t* config
);


 void     cosyvoice_get_generation_config(
    cosyvoice_context_t            ctx,
    cosyvoice_generation_config_t* config
);


 bool     cosyvoice_set_generation_config(
    cosyvoice_context_t                  ctx,
    const cosyvoice_generation_config_t* config
);


 uint32_t cosyvoice_get_sample_rate(cosyvoice_context_t ctx);


 void cosyvoice_set_sampler(cosyvoice_context_t ctx, cosyvoice_sampler_t sampler, void* sampler_ctx);


 void cosyvoice_get_sampler(cosyvoice_context_t ctx, cosyvoice_sampler_t* sampler, void** sampler_ctx);


 cosyvoice_builtin_sampler_rng_policy_t cosyvoice_get_builtin_sampler_rng_policy(cosyvoice_context_t ctx);


 bool cosyvoice_set_builtin_sampler_rng_policy(cosyvoice_context_t ctx, cosyvoice_builtin_sampler_rng_policy_t policy);


 bool cosyvoice_set_sampler_seed(cosyvoice_context_t ctx, uint32_t seed);


 uint32_t cosyvoice_get_sampler_seed(cosyvoice_context_t ctx);


 cosyvoice_prompt_speech_t cosyvoice_prompt_speech_load_from_file(const char* filename);


 bool cosyvoice_prompt_speech_save_to_file(cosyvoice_prompt_speech_t prompt_speech, const char* filename);


 cosyvoice_prompt_t        cosyvoice_prompt_init_from_prompt_speech(cosyvoice_context_t ctx, cosyvoice_prompt_speech_t prompt_speech);


 void cosyvoice_prompt_speech_free(cosyvoice_prompt_speech_t prompt_speech);

 void cosyvoice_prompt_free(cosyvoice_prompt_t prompt);



#define COSYVOICE_TTS_FLAG_TEXT_NORMALIZATION (1u << 0)
#define COSYVOICE_TTS_FLAG_SPLIT_TEXT         (1u << 1)
#define COSYVOICE_TTS_FLAG_FAST_SPLIT         (1u << 2)
#define COSYVOICE_TTS_FLAG_FADE_IN            (1u << 3)

 cosyvoice_tts_context_t cosyvoice_tts_context_new(cosyvoice_context_t ctx, cosyvoice_prompt_t prompt);


 void                    cosyvoice_tts_context_free(cosyvoice_tts_context_t ctx);


 void cosyvoice_tts_context_set_prompt(cosyvoice_tts_context_t ctx, cosyvoice_prompt_t prompt);


 bool cosyvoice_tts_context_set_text_normalization_enabled(cosyvoice_tts_context_t ctx, bool enabled);


 bool cosyvoice_tts_context_get_text_normalization_enabled(cosyvoice_tts_context_t ctx);


 bool cosyvoice_tts_context_set_split_text_enabled(cosyvoice_tts_context_t ctx, bool enabled);


 bool cosyvoice_tts_context_get_split_text_enabled(cosyvoice_tts_context_t ctx);


 bool cosyvoice_tts_context_set_fast_split_text_enabled(cosyvoice_tts_context_t ctx, bool enabled);


 bool cosyvoice_tts_context_get_fast_split_text_enabled(cosyvoice_tts_context_t ctx);


 bool cosyvoice_tts_context_set_fade_in_enabled(cosyvoice_tts_context_t ctx, bool enabled);


 bool cosyvoice_tts_context_get_fade_in_enabled(cosyvoice_tts_context_t ctx);


 uint32_t cosyvoice_tts_context_get_flags(cosyvoice_tts_context_t ctx);


 uint32_t cosyvoice_tts_context_set_flags(cosyvoice_tts_context_t ctx, uint32_t flags);


 bool cosyvoice_tts_zero_shot(
    cosyvoice_tts_context_t        ctx,
    const char*                    text,
    float                          speed,
    cosyvoice_generated_speech_ptr result
);

 bool cosyvoice_tts_instruct(
    cosyvoice_tts_context_t        ctx,
    const char*                    text,
    const char*                    instruction,
    float                          speed,
    cosyvoice_generated_speech_ptr result
);


 bool cosyvoice_tts_cross_lingual(
    cosyvoice_tts_context_t        ctx,
    const char*                    text,
    float                          speed,
    cosyvoice_generated_speech_ptr result
);


 bool cosyvoice_save_wav(const char* filename, const float* data, uint32_t data_len, uint32_t sample_rate);


typedef struct cosyvoice_memory_usage
{
    size_t parameters;        
    size_t kv_cache;          
    size_t token2wav;         
    size_t buffers;           
    size_t cpu_buffers;       
    size_t offloaded_kv_cache;
    size_t random_noise;      
} cosyvoice_memory_usage_t;


 void cosyvoice_get_memory_usage(cosyvoice_context_t ctx, cosyvoice_memory_usage_t* usage);


 void cosyvoice_get_total_memory_usage(cosyvoice_context_t ctx, cosyvoice_memory_usage_t* usage);


 void cosyvoice_empty_buffer_cache(cosyvoice_context_t ctx);

#ifdef __cplusplus
}
#endif

#endif // COSYVOICE_H
