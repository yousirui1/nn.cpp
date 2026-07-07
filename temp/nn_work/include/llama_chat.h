#ifndef __LLAMA_CHAT_H__
#define __LLAMA_CHAT_H__

#include "base.h"
#include <string>


EXPORT void free_llama_chat(void *handle);
EXPORT void* llama_chat_alloc(const char* model_path, int ngl, int n_ctx);
EXPORT int llama_chat_inference(void *handle, std::string system, std::string user, std::string &response);

#endif //__LLAMA_CHAT_H__
