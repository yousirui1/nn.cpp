#include "base.h"
#include "llama.h"
#include "llama_chat.h"
#include <clocale>
#include <cstdio>
#include <cstring>
#include <iostream>
#include <string>
#include <vector>

struct llama_chat_context_t
{
    std::vector<llama_chat_message> history_messages;
    llama_context* ctx;
    llama_sampler* smpl;
    llama_model* model;
    llama_context_params ctx_params;
};

void* llama_chat_alloc(const char *model_path, int ngl, int n_ctx)
{
    llama_chat_context_t* context = new llama_chat_context_t;
 // only print errors
    llama_log_set([](enum ggml_log_level level, const char * text, void * /* user_data */) {
        if (level >= GGML_LOG_LEVEL_ERROR) {
            fprintf(stderr, "%s", text);
        }
    }, nullptr);

    // load dynamic backends
    ggml_backend_load_all();

    // initialize the model
    llama_model_params model_params = llama_model_default_params();
    model_params.n_gpu_layers = ngl;

    context->model = llama_model_load_from_file(model_path, model_params);
    if (!context->model) {
        fprintf(stderr , "%s: error: unable to load model\n" , __func__);
        return nullptr;
    }


    // initialize the context
    llama_context_params ctx_params = llama_context_default_params();
    ctx_params.n_ctx = n_ctx;
    ctx_params.n_batch = n_ctx;

    context->ctx_params = ctx_params;

    context->ctx = llama_init_from_model(context->model, ctx_params);
    if (!context->ctx)
    {
        fprintf(stderr , "%s: error: failed to create the llama_context\n" , __func__);
        return nullptr;
    }

    // initialize the sampler
    context->smpl = llama_sampler_chain_init(llama_sampler_chain_default_params());
    llama_sampler_chain_add(context->smpl, llama_sampler_init_min_p(0.05f, 1));
    llama_sampler_chain_add(context->smpl, llama_sampler_init_temp(0.8f));
    llama_sampler_chain_add(context->smpl, llama_sampler_init_dist(LLAMA_DEFAULT_SEED));
    return context;
}

static std::string llama_chat_generate(llama_chat_context_t* context, const std::string & prompt)
{
    std::string response;
    const bool is_first = llama_memory_seq_pos_max(llama_get_memory(context->ctx), 0) == -1;

    const llama_vocab* vocab  = llama_model_get_vocab(context->model);

    // tokenize the prompt
    const int n_prompt_tokens = -llama_tokenize(vocab, prompt.c_str(), prompt.size(), NULL, 0, is_first, true);
    std::vector<llama_token> prompt_tokens(n_prompt_tokens);
    if (llama_tokenize(vocab, prompt.c_str(), prompt.size(), prompt_tokens.data(), prompt_tokens.size(), is_first, true) < 0) {
        GGML_ABORT("failed to tokenize the prompt\n");
    }

    // prepare a batch for the prompt
    llama_batch batch = llama_batch_get_one(prompt_tokens.data(), prompt_tokens.size());
    llama_token new_token_id;
    while (true) 
    {
        // check if we have enough space in the context to evaluate this batch
        int n_ctx = llama_n_ctx(context->ctx);
        int n_ctx_used = llama_memory_seq_pos_max(llama_get_memory(context->ctx), 0) + 1;
        if (n_ctx_used + batch.n_tokens > n_ctx) {
            printf("\033[0m\n");
            fprintf(stderr, "context size exceeded\n");
            exit(0);
        }

        int ret = llama_decode(context->ctx, batch);
        if (ret != 0) {
            GGML_ABORT("failed to decode, ret = %d\n", ret);
        }

        // sample the next token
        new_token_id = llama_sampler_sample(context->smpl, context->ctx, -1);

        // is it an end of generation?
        if (llama_vocab_is_eog(vocab, new_token_id)) {
            break;
        }

        // convert the token to a string, print it and add it to the response
        char buf[256];
        int n = llama_token_to_piece(vocab, new_token_id, buf, sizeof(buf), 0, true);
        if (n < 0) {
            GGML_ABORT("failed to convert token to piece\n");
        }
        std::string piece(buf, n);
        //cb to do 
        //
        printf("%s", piece.c_str());
        fflush(stdout);
        response += piece;
        // prepare the next batch with the sampled token
        batch = llama_batch_get_one(&new_token_id, 1);
    }
    return response;
}


int llama_chat_inference(void *handle, std::string system, std::string user, std::string & response)
{
    llama_chat_context_t* context = (llama_chat_context_t*) handle;

    // helper function to evaluate a prompt and generate a response

    std::vector<llama_chat_message> messages;
    std::vector<char> formatted(llama_n_ctx(context->ctx));
    int prev_len = 0;

    std::string user_prompt = system + "{" + user + "}";
    
    const char * tmpl = llama_model_chat_template(context->model, /* name */ nullptr);

    // add the user input to the message list and format it
    messages.push_back({"user", strdup(user_prompt.c_str())});

    int new_len = llama_chat_apply_template(tmpl, messages.data(), messages.size(), true, formatted.data(), formatted.size());
    if (new_len > (int)formatted.size()) 
    {
        formatted.resize(new_len);
        new_len = llama_chat_apply_template(tmpl, messages.data(), messages.size(), true, formatted.data(), formatted.size());
    }
    if (new_len < 0) {
        fprintf(stderr, "failed to apply the chat template\n");
        return 1;
    }

    // remove previous messages to obtain the prompt to generate the response
    std::string prompt(formatted.begin() + prev_len, formatted.begin() + new_len);

    // generate a response
    response = llama_chat_generate(context, prompt);
    //LOG_DEBUG("response %s", response.c_str());

    // add the response to the messages
    messages.push_back({"assistant", strdup(response.c_str())});
    prev_len = llama_chat_apply_template(tmpl, messages.data(), messages.size(), false, nullptr, 0);
    if (prev_len < 0) 
    {
        fprintf(stderr, "failed to apply the chat template\n");
        return ERROR;
    }
    //callback

    for (auto & msg : messages) 
    {
        free(const_cast<char *>(msg.content));
    }
    return SUCCESS;
}

void free_llama_chat(void *handle)
{
    llama_chat_context_t* context = (llama_chat_context_t*) handle;

    // free resources
    llama_sampler_free(context->smpl);
    llama_free(context->ctx);
    llama_model_free(context->model);
    delete context;
}

/*
 * ||中文	zh      |英语	en      |法语	fr   |葡萄牙语	pt      | 西班牙语	es  ||
   ||日语	ja      |土耳其语 tr    |俄语	ru   |阿拉伯语	ar      | 韩语	ko	    ||
   ||泰语	th      |意大利语 it	|德语	de   |越南语	vi	    |马来语	ms	    ||
   ||印尼语	id      |菲律宾语 tl	|印地语	hi   |繁体中文 zh-Hant  |波兰语	pl	    ||
   ||捷克语	cs      |荷兰语	nl	    |高棉语 km   |缅甸语	my	    |波斯语	fa      ||
   ||古吉拉特语	gu  |乌尔都语	ur  |泰卢固语 te |马拉地语 mr       |希伯来语	he  ||
   ||孟加拉语	bn  |泰米尔语	ta  |乌克兰语 uk |藏语	bo	        | 哈萨克语	kk  ||
   ||蒙古语	mn      |维吾尔语 ug    | 粤语	yue	 ||
*/

#if 0
int main(int argc, char *argv[])
{
    const char *model_path = "/home/ysr/project/models/gguf/HY-MT1.5-1.8B-Q8_0.gguf";
    struct llama_chat_context_t *context = llama_chat_alloc(model_path, 100, 2048);
    //zh en fr pt ar 
    const char *system_prompt = "Translate the following segment into {ar}, without additional explanation.";
    llama_chat_inference(context, system_prompt, "讲一个笑话");
    free_llama_chat(context);
}
#endif
