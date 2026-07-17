#pragma once

#include <stdint.h>

#ifdef __GNUC__
#define EXPORT __attribute__((visibility("default")))
#elif _WIN32
#define EXPORT __declspec(dllimport)
#endif

#ifdef __cplusplus
extern "C" {
#else
	#include <stdbool.h>
#endif

EXPORT bool load_audio_file(const char* filename, float** data, uint32_t* length, uint32_t* sample_rate);

bool audio_resample(
	const float* input,
	uint32_t     input_length,
	uint32_t     input_sample_rate,
	float**      output,
	uint32_t*    output_length,
	uint32_t     output_sample_rate
);

EXPORT bool save_audio_file(const char* filename, const float* data, uint32_t length, uint32_t sample_rate
);

EXPORT void free_audio_data(float* data);

#ifdef __cplusplus
}
#endif

