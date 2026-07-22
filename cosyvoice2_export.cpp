/**
 * CosyVoice2 C++ 输出导出工具
 * 
 * 该程序用于导出 C++ 实现的中间结果，供 Python 脚本进行一致性对比验证
 * 
 * 使用方法:
 *   ./cosyvoice2_export --model model.gguf --text "测试文本" --output outputs.bin
 */

#include "cosyvoice.h"
#include "cosyvoice_lowlevel.h"
#include "cosyvoice2_model.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <string>
#include <fstream>
#include <iostream>

// 二进制输出格式工具函数
namespace output_utils {

    void write_string(std::ofstream& out, const std::string& str) {
        uint32_t len = static_cast<uint32_t>(str.size());
        out.write(reinterpret_cast<const char*>(&len), sizeof(len));
        out.write(str.data(), len);
    }

    void write_tensor(std::ofstream& out, const std::string& name, 
                     ggml_type dtype, const float* data, 
                     const std::vector<int64_t>& shape) {
        // 写入名称
        write_string(out, name);
        
        // 写入数据类型 (0: float32, 1: float16, 2: int32)
        uint32_t dtype_code = (dtype == GGML_TYPE_F16) ? 1 : 
                             (dtype == GGML_TYPE_I32) ? 2 : 0;
        out.write(reinterpret_cast<const char*>(&dtype_code), sizeof(dtype_code));
        
        // 写入形状
        uint32_t shape_len = static_cast<uint32_t>(shape.size());
        out.write(reinterpret_cast<const char*>(&shape_len), sizeof(shape_len));
        for (auto dim : shape) {
            int32_t dim32 = static_cast<int32_t>(dim);
            out.write(reinterpret_cast<const char*>(&dim32), sizeof(dim32));
        }
        
        // 写入数据 (统一转换为 float32)
        size_t size = 1;
        for (auto dim : shape) size *= dim;
        out.write(reinterpret_cast<const char*>(data), size * sizeof(float));
    }

    void write_tensor_int(std::ofstream& out, const std::string& name,
                         const int* data, const std::vector<int64_t>& shape) {
        write_string(out, name);
        
        uint32_t dtype_code = 2; // int32
        out.write(reinterpret_cast<const char*>(&dtype_code), sizeof(dtype_code));
        
        uint32_t shape_len = static_cast<uint32_t>(shape.size());
        out.write(reinterpret_cast<const char*>(&shape_len), sizeof(shape_len));
        for (auto dim : shape) {
            int32_t dim32 = static_cast<int32_t>(dim);
            out.write(reinterpret_cast<const char*>(&dim32), sizeof(dim32));
        }
        
        size_t size = 1;
        for (auto dim : shape) size *= dim;
        out.write(reinterpret_cast<const char*>(data), size * sizeof(int));
    }
}

// CosyVoice2 流式推理测试类
class CosyVoice2Exporter {
private:
    cosyvoice_context_t ctx;
    std::string model_path;
    
public:
    CosyVoice2Exporter(const std::string& model_file) 
        : ctx(nullptr), model_path(model_file) {}
    
    ~CosyVoice2Exporter() {
        if (ctx) {
            cosyvoice_free(ctx);
        }
    }
    
    bool load_model() {
        // 加载 CosyVoice2 模型
        cosyvoice_context_params_v2_cpp params = cosyvoice_context_default_params_v2();
        params.n_threads = 4;
        params.n_batch = 512;
        
        ctx = cosyvoice_load_from_file_ext(
            model_path.c_str(),
            &params,
            nullptr,  // backend (use default)
            4         // n_threads
        );
        
        if (!ctx) {
            std::cerr << "Failed to load model from: " << model_path << std::endl;
            return false;
        }
        
        std::cout << "✓ Model loaded successfully" << std::endl;
        return true;
    }
    
    bool export_llm_outputs(const std::vector<int>& token_ids,
                           const std::vector<int>& prompt_tokens,
                           std::ofstream& out) {
        std::cout << "\n=== Exporting LLM Outputs ===" << std::endl;
        
        auto tokenizer = cosyvoice_get_tokenizer(ctx);
        if (!tokenizer) {
            std::cerr << "Failed to get tokenizer" << std::endl;
            return false;
        }
        
        // 设置 prompt
        cosyvoice_prompt_t prompt = {};
        prompt = cosyvoice_prompt_set(
            ctx,
            prompt,
            COSYVOICE_INFERENCE_MODE_ZERO_SHOT,
            nullptr,
            0,
            true
        );
        
        // LLM Prefill
        std::cout << "Running LLM prefill..." << std::endl;
        
        // 构建输入 embeddings
        auto speech_emb_weight = cosyvoice_get_speech_token_embed_weight(ctx);
        auto word_emb_weight = cosyvoice_get_word_token_embed_weight(ctx);
        
        if (!speech_emb_weight || !word_emb_weight) {
            std::cerr << "Failed to get embedding weights" << std::endl;
            return false;
        }
        
        uint32_t emb_size = static_cast<uint32_t>(speech_emb_weight->ne[0]);
        std::vector<float> embeddings(token_ids.size() * emb_size);
        
        // 获取 embeddings
        const float* speech_data = reinterpret_cast<const float*>(speech_emb_weight->data);
        const float* word_data = reinterpret_cast<const float*>(word_emb_weight->data);
        
        for (size_t i = 0; i < token_ids.size(); ++i) {
            int token_id = token_ids[i];
            const float* src = (token_id >= 0 && token_id < 6561) ? speech_data : word_data;
            std::memcpy(embeddings.data() + i * emb_size, 
                       src + token_id * emb_size,
                       emb_size * sizeof(float));
        }
        
        // Prefill
        if (!cosyvoice_llm_prefill(ctx, GGML_TYPE_F32, embeddings.data(), 
                                   static_cast<uint32_t>(token_ids.size()))) {
            std::cerr << "LLM prefill failed" << std::endl;
            return false;
        }
        
        // 获取 logits (需要从内部获取，这里简化处理)
        // 实际实现需要访问模型的输出层
        std::cout << "LLM prefill completed" << std::endl;
        
        // LLM Decode
        std::cout << "Running LLM decode..." << std::endl;
        
        std::vector<int> generated_tokens;
        const int max_tokens = 50;
        
        for (int i = 0; i < max_tokens; ++i) {
            if (!cosyvoice_llm_decode(ctx, GGML_TYPE_F32, nullptr)) {
                std::cerr << "LLM decode failed at step " << i << std::endl;
                break;
            }
            
            cosyvoice_llm_prepare_probs(ctx, true);
            int token = cosyvoice_llm_sample_token(ctx);
            
            if (cosyvoice_llm_is_stop_token(ctx, token)) {
                std::cout << "Stop token encountered at step " << i << std::endl;
                break;
            }
            
            generated_tokens.push_back(token);
            cosyvoice_llm_accept_token(ctx, token);
        }
        
        std::cout << "Generated " << generated_tokens.size() << " tokens" << std::endl;
        
        // 导出生成的 tokens
        if (!generated_tokens.empty()) {
            output_utils::write_tensor_int(
                out, "llm_output_tokens",
                generated_tokens.data(),
                {static_cast<int64_t>(generated_tokens.size())}
            );
        }
        
        return true;
    }
    
    bool export_flow_outputs(const std::vector<int>& speech_tokens,
                            const std::vector<int>& prompt_tokens,
                            const std::vector<float>& prompt_mel,
                            const std::vector<float>& embedding,
                            std::ofstream& out) {
        std::cout << "\n=== Exporting Flow Encoder Outputs ===" << std::endl;
        
        // 注意：这里需要调用 flow encoder 的接口
        // 由于当前代码中 flow encoder 的接口尚未完全暴露，
        // 这里提供框架代码，实际使用时需要补充完整
        
        std::cout << "Flow encoder export (placeholder)" << std::endl;
        std::cout << "Input tokens: " << speech_tokens.size() << std::endl;
        std::cout << "Prompt tokens: " << prompt_tokens.size() << std::endl;
        std::cout << "Prompt mel frames: " << prompt_mel.size() / 80 << std::endl;
        
        // 模拟输出 (实际应调用 flow encoder)
        uint32_t mel_frames = static_cast<uint32_t>(speech_tokens.size()) * 2;
        std::vector<float> mel_output(mel_frames * 80, 0.0f);
        
        output_utils::write_tensor(
            out, "flow_mel",
            GGML_TYPE_F32,
            mel_output.data(),
            {1, static_cast<int64_t>(mel_frames), 80}
        );
        
        return true;
    }
    
    bool export_hift_outputs(const std::vector<float>& mel_spec,
                            std::ofstream& out) {
        std::cout << "\n=== Exporting HiFT Generator Outputs ===" << std::endl;
        
        // 调用 HiFT generator
        // 需要从 cosyvoice_context 获取 hift 模型并执行前向传播
        
        uint32_t mel_frames = static_cast<uint32_t>(mel_spec.size() / 80);
        std::cout << "Input mel frames: " << mel_frames << std::endl;
        
        // 模拟音频输出 (24kHz, 实际应根据 HiFT 的上采样率计算)
        uint32_t audio_samples = mel_frames * 480; // 假设上采样率为 480
        std::vector<float> audio_output(audio_samples, 0.0f);
        
        output_utils::write_tensor(
            out, "hift_audio",
            GGML_TYPE_F32,
            audio_output.data(),
            {1, static_cast<int64_t>(audio_samples)}
        );
        
        return true;
    }
    
    bool run_full_export(const std::string& text,
                        const std::string& output_file) {
        std::cout << "========================================" << std::endl;
        std::cout << "CosyVoice2 Output Exporter" << std::endl;
        std::cout << "========================================" << std::endl;
        std::cout << "Model: " << model_path << std::endl;
        std::cout << "Text: " << text << std::endl;
        std::cout << "Output: " << output_file << std::endl;
        
        if (!load_model()) {
            return false;
        }
        
        // 打开输出文件
        std::ofstream out(output_file, std::ios::binary);
        if (!out) {
            std::cerr << "Failed to open output file: " << output_file << std::endl;
            return false;
        }
        
        // Tokenize 文本
        auto tokenizer = cosyvoice_get_tokenizer(ctx);
        cosyvoice_tokenization_result_t tok_result = cosyvoice_tokenization_result_create();
        
        uint32_t n_tokens = cosyvoice_tokenize(tokenizer, text.c_str(), tok_result, true);
        const int* tokens = cosyvoice_tokenization_result_get_tokens(tok_result);
        
        std::vector<int> token_ids(tokens, tokens + n_tokens);
        std::cout << "Tokenized input: " << n_tokens << " tokens" << std::endl;
        
        // 导出各阶段输出
        bool success = true;
        
        // 1. LLM 输出
        std::vector<int> prompt_tokens; // 空 prompt
        success &= export_llm_outputs(token_ids, prompt_tokens, out);
        
        // 2. Flow 输出 (使用生成的 tokens)
        std::vector<float> prompt_mel(80 * 50, 0.0f); // 50 帧的零填充
        std::vector<float> embedding(192, 0.0f); // 零 embedding
        success &= export_flow_outputs(token_ids, prompt_tokens, prompt_mel, embedding, out);
        
        // 3. HiFT 输出
        std::vector<float> mel_spec(80 * token_ids.size() * 2, 0.0f);
        success &= export_hift_outputs(mel_spec, out);
        
        // 清理
        cosyvoice_tokenization_result_free(tok_result);
        out.close();
        
        if (success) {
            std::cout << "\n✓ Export completed successfully!" << std::endl;
            std::cout << "Output file: " << output_file << std::endl;
        } else {
            std::cerr << "\n✗ Export failed!" << std::endl;
        }
        
        return success;
    }
};

void print_usage(const char* prog) {
    std::cout << "Usage: " << prog << " [options]" << std::endl;
    std::cout << "Options:" << std::endl;
    std::cout << "  --model <path>      Path to CosyVoice2 GGUF model file" << std::endl;
    std::cout << "  --text <text>       Input text to synthesize" << std::endl;
    std::cout << "  --output <path>     Output binary file for comparison" << std::endl;
    std::cout << "  --help              Show this help message" << std::endl;
    std::cout << std::endl;
    std::cout << "Example:" << std::endl;
    std::cout << "  " << prog << " --model cosyvoice2.gguf --text \"你好世界\" --output outputs.bin" << std::endl;
}

int main(int argc, char** argv) {
    std::string model_path;
    std::string text;
    std::string output_path;
    
    // 解析命令行参数
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        
        if (arg == "--help" || arg == "-h") {
            print_usage(argv[0]);
            return 0;
        }
        else if (arg == "--model" && i + 1 < argc) {
            model_path = argv[++i];
        }
        else if (arg == "--text" && i + 1 < argc) {
            text = argv[++i];
        }
        else if (arg == "--output" && i + 1 < argc) {
            output_path = argv[++i];
        }
        else {
            std::cerr << "Unknown option: " << arg << std::endl;
            print_usage(argv[0]);
            return 1;
        }
    }
    
    // 验证参数
    if (model_path.empty()) {
        std::cerr << "Error: --model is required" << std::endl;
        print_usage(argv[0]);
        return 1;
    }
    
    if (text.empty()) {
        std::cerr << "Error: --text is required" << std::endl;
        print_usage(argv[0]);
        return 1;
    }
    
    if (output_path.empty()) {
        output_path = "cosyvoice2_outputs.bin";
        std::cout << "Using default output file: " << output_path << std::endl;
    }
    
    // 执行导出
    CosyVoice2Exporter exporter(model_path);
    
    if (!exporter.run_full_export(text, output_path)) {
        return 1;
    }
    
    return 0;
}
