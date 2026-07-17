#ifndef COSYVOICE_FRONTEND_H
#define COSYVOICE_FRONTEND_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif


typedef struct cosyvoice_text
{
	uint32_t    length; 
	const char* text; 
}* cosyvoice_text_ptr;


cosyvoice_text_ptr cosyvoice_text_create();


void cosyvoice_text_free(cosyvoice_text_ptr text);

void cosyvoice_frontend_util_text_normalize(
	const char* text,
	uint32_t    text_len,
	const char* locale,
	cosyvoice_text_ptr normalized_text
);

typedef struct cosyvoice_frontend_context* cosyvoice_frontend_context_t;

cosyvoice_frontend_context_t cosyvoice_frontend_load_from_files(
	const char* speech_tokenizer,
	const char* campplus
);

struct OrtEnv;
struct OrtSessionOptions;

cosyvoice_frontend_context_t cosyvoice_frontend_load(
	const void* speech_tokenizer_data,
	size_t speech_tokenizer_size,
	const void* campplus_data,
	size_t campplus_size,
	const struct OrtEnv* env,
	const struct OrtSessionOptions* session_options
);

cosyvoice_prompt_speech_t cosyvoice_frontend_prompt_speech(
	cosyvoice_frontend_context_t ctx,
	float* speech,
	uint32_t speech_len,
	uint32_t sample_rate,
	const char* prompt_text
);

cosyvoice_prompt_speech_t cosyvoice_frontend_prompt_speech_direct(
	cosyvoice_frontend_context_t ctx,
	float* speech_16k,
	uint32_t speech_16k_len,
	float* speech_24k,
	uint32_t speech_24k_len,
	const char* prompt_text
);

void cosyvoice_frontend_free(cosyvoice_frontend_context_t ctx);

#ifdef __cplusplus
}
#endif

#endif // COSYVOICE_FRONTEND_H
