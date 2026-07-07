#pragma once




struct cosyvoice_model_context
{
    // Configuration
    virtual uint32_t get_sample_rate() = 0; 
    virtual void get_default_generation_config(cosyvoice_generation_config_t* config) = 0; 
    virtual void get_generation_config(cosyvoice_generation_config_t* config) = 0; 
    virtual bool set_generation_config(const cosyvoice_generation_config_t* config) = 0; 
    virtual void get_context_params(cosyvoice_context_params_t* params) = 0;
    virtual const char* get_instruction_prefix() = 0;
    virtual const char* get_architecture() = 0; 
    virtual bool is_backend_uma() = 0;
    virtual bool set_worker_no(uint32_t worker_no) = 0; 
    virtual uint32_t get_worker_no() = 0; 
    virtual uint32_t get_n_workers() = 0; 

    // Sampling Configuration
    virtual void get_sampler(cosyvoice_sampler_t* sampler, void** sampler_ctx) = 0;
    virtual void set_sampler(cosyvoice_sampler_t sampler, void* sampler_ctx) = 0;
    virtual cosyvoice_builtin_sampler_rng_policy_t get_builtin_sampler_rng_policy() = 0; 
    virtual bool set_builtin_sampler_rng_policy(cosyvoice_builtin_sampler_rng_policy_t policy) = 0;
    virtual bool set_sampler_seed(uint32_t seed) = 0; 
    virtual uint32_t get_sampler_seed() = 0; 

    // LLM Operations
    virtual bool llm_prefill(ggml_type type, const void* data, uint32_t seq_len) = 0;
    virtual bool llm_decode(ggml_type type, const void* data) = 0; 
    virtual void llm_prepare_probs(bool allow_stop_tokens) = 0; 

	// Embedding Access
	virtual const ggml_tensor* get_word_token_embed_weight() = 0; 
	virtual const ggml_tensor* get_speech_token_embed_weight() = 0; 

	// KV Cache
	virtual uint32_t llm_get_kv_cache_len() = 0; 
	virtual bool llm_set_kv_cache_len(uint32_t len) = 0;

	// Token Sampling and Acceptance
	virtual int llm_sample_token() = 0; 
	virtual bool llm_is_stop_token(int token_id) = 0; 
	virtual void llm_accept_token(int token_id) = 0; 
	virtual void llm_clear_accepted_tokens() = 0;
	virtual uint32_t llm_get_n_accepted_tokens() = 0;
	virtual const int* llm_get_accepted_tokens() = 0; 

	// Inference
	virtual bool llm_job(const int* text, uint32_t text_len, cosyvoice_prompt_t prompt) = 0; 
	virtual bool token2wav(
		const int*                 token_ids,
		uint32_t                   n_tokens,
		float                      speed,
		cosyvoice_prompt_t         prompt,
		cosyvoice_generated_speech_ptr result
	) = 0;

	// Status
	virtual ggml_status get_last_status() = 0;

	// Prompt Management
	virtual void set_prompt(
		cosyvoice_prompt_t         prompt,
		cosyvoice_inference_mode_t mode,
		const int*                 instruction,
		uint32_t                   instruction_length
	) = 0;

    // Memory
    virtual void get_memory_usage(cosyvoice_memory_usage_t* usage) = 0;
    virtual void get_total_memory_usage(cosyvoice_memory_usage_t* usage) = 0;
    virtual void empty_buffer_cache() = 0; 

	// Noise Control
	virtual void set_noise_callback(cosyvoice_noise_callback_t callback, void* callback_ctx) = 0; 
	virtual void get_noise_callback(cosyvoice_noise_callback_t* callback, void** callback_ctx) = 0;
	virtual uint32_t get_hift_rand_ini_len() = 0;
	virtual void set_hift_rand_ini(const float* data) = 0;
};



struct cosyvoice_tokenization_result
{
	virtual int* get_tokens() = 0;
	virtual uint32_t get_n_tokens() = 0;
};


struct cosyvoice_tokenizer_context
{
	virtual uint32_t tokenize(const char* text, uint32_t text_len, cosyvoice_tokenization_result_t result, bool parse_special = true) = 0;

	inline uint32_t tokenize(const char* text, cosyvoice_tokenization_result_t result, bool parse_special = true)
	{
		auto p = text;
		while (*p) ++p;
		return tokenize(text, static_cast<uint32_t>(p - text), result, parse_special);
	}
};


struct cosyvoice_context : virtual cosyvoice_model_context,
                           virtual cosyvoice_tokenizer_context {};
