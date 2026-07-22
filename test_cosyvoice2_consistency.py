#!/usr/bin/env python3
"""
CosyVoice2 C++ vs Python 输出一致性测试脚本

该脚本用于验证 C++ 实现的 CosyVoice2 流式推理与官方 Python 实现的输出一致性。
参考：https://github.com/FunAudioLLM/CosyVoice.git

主要测试内容：
1. LLM prefill/decode 输出对比
2. Flow encoder 的 mel spectrogram 对比
3. HiFT generator 的音频波形对比
4. 流式推理的分块输出一致性
"""

import os
import sys
import torch
import numpy as np
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import time

# 添加 CosyVoice 路径
COSYVOICE_REPO_PATH = os.environ.get('COSYVOICE_REPO_PATH', '/tmp/CosyVoice')
if os.path.exists(COSYVOICE_REPO_PATH):
    sys.path.insert(0, COSYVOICE_REPO_PATH)

try:
    from cosyvoice.flow.flow import CausalMaskedDiffWithXvec
    from cosyvoice.flow.flow_matching import CausalConditionalCFM
    from cosyvoice.hift.generator import CausalHiFTGenerator
    from cosyvoice.llm.llm import CosyVoice2LM
    PYTHON_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Cannot import CosyVoice Python modules: {e}")
    print("Please install CosyVoice: pip install -e .")
    PYTHON_AVAILABLE = False


class CosyVoice2Comparator:
    """CosyVoice2 C++ vs Python 输出对比器"""
    
    def __init__(self, model_dir: str, device: str = 'cpu'):
        self.device = torch.device(device)
        self.model_dir = Path(model_dir)
        self.python_model = None
        self.cpp_output = None
        self.tolerance = {
            'llm_logits': 1e-4,
            'flow_mel': 1e-3,
            'hift_audio': 1e-2,
            'streaming_chunk': 5e-3
        }
        
    def load_python_model(self, config_path: str):
        """加载 Python 参考模型"""
        if not PYTHON_AVAILABLE:
            raise RuntimeError("Python CosyVoice module not available")
            
        from hyperpyyaml import load_hyperpyyaml
        
        with open(config_path, 'r', encoding='utf-8') as f:
            configs = load_hyperpyyaml(f)
        
        self.python_model = {
            'flow': configs['flow'].to(self.device).eval(),
            'hift': configs['hift'].to(self.device).eval(),
            'llm': configs['llm'].to(self.device).eval() if 'llm' in configs else None
        }
        
        # 设置 eval 模式并禁用梯度
        for name, model in self.python_model.items():
            if model is not None:
                model.eval()
                for param in model.parameters():
                    param.requires_grad = False
                    
        print(f"✓ Python models loaded successfully")
        
    def set_cpp_output(self, output_dict: Dict[str, np.ndarray]):
        """设置 C++ 实现输出（从二进制文件或共享内存读取）"""
        self.cpp_output = output_dict
        print(f"✓ C++ outputs loaded: {list(output_dict.keys())}")
        
    def compare_llm_prefill(
        self,
        token_ids: List[int],
        prompt_tokens: Optional[List[int]] = None,
        max_relative_error: float = 1e-3
    ) -> Dict:
        """
        对比 LLM prefill 阶段的输出
        
        Args:
            token_ids: 输入文本 token IDs
            prompt_tokens: 可选的 prompt tokens
            max_relative_error: 最大允许的相对误差
            
        Returns:
            对比结果字典
        """
        result = {
            'passed': True,
            'max_abs_error': 0.0,
            'max_rel_error': 0.0,
            'shape_match': True,
            'details': ''
        }
        
        if self.python_model is None or self.cpp_output is None:
            result['passed'] = False
            result['details'] = "Models not loaded"
            return result
        
        # Python 前向传播
        with torch.no_grad():
            py_logits = self._python_llm_prefill(token_ids, prompt_tokens)
        
        # C++ 输出
        cpp_logits = self.cpp_output.get('llm_logits', None)
        
        if cpp_logits is None:
            result['passed'] = False
            result['details'] = "C++ logits not found"
            return result
        
        # 转换为 numpy 进行对比
        py_logits_np = py_logits.cpu().numpy()
        
        # 形状检查
        if py_logits_np.shape != cpp_logits.shape:
            result['passed'] = False
            result['shape_match'] = False
            result['details'] = f"Shape mismatch: Python {py_logits_np.shape} vs C++ {cpp_logits.shape}"
            return result
        
        # 计算误差
        abs_error = np.abs(py_logits_np - cpp_logits)
        rel_error = np.zeros_like(abs_error)
        mask = np.abs(py_logits_np) > 1e-8
        rel_error[mask] = abs_error[mask] / np.abs(py_logits_np[mask])
        
        result['max_abs_error'] = float(np.max(abs_error))
        result['max_rel_error'] = float(np.max(rel_error))
        result['mean_abs_error'] = float(np.mean(abs_error))
        
        if result['max_rel_error'] > max_relative_error:
            result['passed'] = False
            result['details'] = f"Relative error {result['max_rel_error']:.6e} > threshold {max_relative_error:.6e}"
        else:
            result['details'] = "✓ Test passed"
            
        return result
    
    def compare_flow_encoder(
        self,
        token_ids: np.ndarray,
        prompt_tokens: np.ndarray,
        prompt_mel: np.ndarray,
        embedding: np.ndarray,
        streaming: bool = False,
        finalize: bool = True
    ) -> Dict:
        """
        对比 Flow Encoder 的 mel spectrogram 输出
        
        Args:
            token_ids: 语音 token IDs [1, T]
            prompt_tokens: prompt tokens [1, T_prompt]
            prompt_mel: prompt mel spectrogram [1, T_mel, 80]
            embedding: speaker embedding [192,]
            streaming: 是否使用流式模式
            finalize: 是否是最后一个 chunk
            
        Returns:
            对比结果字典
        """
        result = {
            'passed': True,
            'max_abs_error': 0.0,
            'max_rel_error': 0.0,
            'shape_match': True,
            'details': ''
        }
        
        if self.python_model is None or self.cpp_output is None:
            result['passed'] = False
            result['details'] = "Models not loaded"
            return result
        
        # 转换输入为 tensor
        token_tensor = torch.from_numpy(token_ids).to(self.device)
        prompt_token_tensor = torch.from_numpy(prompt_tokens).to(self.device)
        prompt_mel_tensor = torch.from_numpy(prompt_mel).to(self.device)
        embedding_tensor = torch.from_numpy(embedding).to(self.device)
        
        # Python 前向传播
        with torch.no_grad():
            py_mel, _ = self.python_model['flow'].inference(
                token=token_tensor,
                token_len=torch.tensor([token_tensor.shape[1]], device=self.device),
                prompt_token=prompt_token_tensor,
                prompt_token_len=torch.tensor([prompt_token_tensor.shape[1]], device=self.device),
                prompt_feat=prompt_mel_tensor,
                prompt_feat_len=torch.tensor([prompt_mel_tensor.shape[1]], device=self.device),
                embedding=embedding_tensor.unsqueeze(0),
                streaming=streaming,
                finalize=finalize
            )
        
        # C++ 输出
        cpp_mel = self.cpp_output.get('flow_mel', None)
        
        if cpp_mel is None:
            result['passed'] = False
            result['details'] = "C++ flow mel not found"
            return result
        
        py_mel_np = py_mel.cpu().numpy()
        
        # 形状检查
        if py_mel_np.shape != cpp_mel.shape:
            result['passed'] = False
            result['shape_match'] = False
            result['details'] = f"Shape mismatch: Python {py_mel_np.shape} vs C++ {cpp_mel.shape}"
            return result
        
        # 计算误差
        abs_error = np.abs(py_mel_np - cpp_mel)
        rel_error = np.zeros_like(abs_error)
        mask = np.abs(py_mel_np) > 1e-6
        rel_error[mask] = abs_error[mask] / np.abs(py_mel_np[mask])
        
        result['max_abs_error'] = float(np.max(abs_error))
        result['max_rel_error'] = float(np.max(rel_error))
        result['mean_abs_error'] = float(np.mean(abs_error))
        
        threshold = self.tolerance['flow_mel']
        if result['max_rel_error'] > threshold:
            result['passed'] = False
            result['details'] = f"Relative error {result['max_rel_error']:.6e} > threshold {threshold:.6e}"
        else:
            result['details'] = "✓ Flow encoder test passed"
            
        return result
    
    def compare_hift_generator(
        self,
        mel_spec: np.ndarray,
        speed: float = 1.0
    ) -> Dict:
        """
        对比 HiFT Generator 的音频输出
        
        Args:
            mel_spec: mel spectrogram [1, 80, T]
            speed: 语速因子
            
        Returns:
            对比结果字典
        """
        result = {
            'passed': True,
            'max_abs_error': 0.0,
            'max_rel_error': 0.0,
            'shape_match': True,
            'snr_db': 0.0,
            'details': ''
        }
        
        if self.python_model is None or self.cpp_output is None:
            result['passed'] = False
            result['details'] = "Models not loaded"
            return result
        
        # 转换输入为 tensor
        mel_tensor = torch.from_numpy(mel_spec).to(self.device)
        
        # Python 前向传播
        with torch.no_grad():
            py_audio, _ = self.python_model['hift'](speech_feat=mel_tensor)
        
        # C++ 输出
        cpp_audio = self.cpp_output.get('hift_audio', None)
        
        if cpp_audio is None:
            result['passed'] = False
            result['details'] = "C++ hift audio not found"
            return result
        
        py_audio_np = py_audio.cpu().numpy()
        
        # 形状检查
        if py_audio_np.shape != cpp_audio.shape:
            result['passed'] = False
            result['shape_match'] = False
            result['details'] = f"Shape mismatch: Python {py_audio_np.shape} vs C++ {cpp_audio.shape}"
            return result
        
        # 计算误差
        abs_error = np.abs(py_audio_np - cpp_audio)
        rel_error = np.zeros_like(abs_error)
        mask = np.abs(py_audio_np) > 1e-6
        rel_error[mask] = abs_error[mask] / np.abs(py_audio_np[mask])
        
        result['max_abs_error'] = float(np.max(abs_error))
        result['max_rel_error'] = float(np.max(rel_error))
        result['mean_abs_error'] = float(np.mean(abs_error))
        
        # 计算 SNR
        signal_power = np.mean(py_audio_np ** 2)
        noise_power = np.mean((py_audio_np - cpp_audio) ** 2)
        if noise_power > 1e-10:
            result['snr_db'] = 10 * np.log10(signal_power / noise_power)
        else:
            result['snr_db'] = float('inf')
        
        threshold = self.tolerance['hift_audio']
        if result['max_rel_error'] > threshold:
            result['passed'] = False
            result['details'] = f"Relative error {result['max_rel_error']:.6e} > threshold {threshold:.6e}, SNR: {result['snr_db']:.2f} dB"
        else:
            result['details'] = f"✓ HiFT test passed, SNR: {result['snr_db']:.2f} dB"
            
        return result
    
    def compare_streaming_inference(
        self,
        chunks: List[Dict],
        final_result: Dict
    ) -> Dict:
        """
        对比流式推理的整体一致性
        
        Args:
            chunks: 每个 chunk 的输入和输出
            final_result: 最终完整输出
            
        Returns:
            对比结果字典
        """
        result = {
            'passed': True,
            'chunk_errors': [],
            'final_error': 0.0,
            'details': ''
        }
        
        # 累积 chunk 输出
        accumulated_py = []
        accumulated_cpp = []
        
        for i, chunk in enumerate(chunks):
            # 这里需要调用相应的对比函数
            # 简化示例，实际需要更详细的实现
            pass
        
        # 对比最终结果
        if 'final_mel' in final_result and 'final_mel_cpp' in final_result:
            py_final = final_result['final_mel']
            cpp_final = final_result['final_mel_cpp']
            
            abs_error = np.abs(py_final - cpp_final)
            result['final_error'] = float(np.max(abs_error))
            
            if result['final_error'] > self.tolerance['streaming_chunk']:
                result['passed'] = False
                result['details'] = f"Final chunk error too large: {result['final_error']:.6e}"
            else:
                result['details'] = "✓ Streaming inference test passed"
        
        return result
    
    def _python_llm_prefill(
        self,
        token_ids: List[int],
        prompt_tokens: Optional[List[int]] = None
    ) -> torch.Tensor:
        """Python LLM prefill 实现"""
        llm = self.python_model['llm']
        if llm is None:
            raise RuntimeError("LLM model not loaded")
        
        # 构建输入
        input_ids = torch.tensor([token_ids], dtype=torch.long, device=self.device)
        
        with torch.no_grad():
            outputs = llm(input_ids, use_cache=False)
            logits = outputs.logits
        
        return logits
    
    def run_all_tests(self, test_data: Dict) -> Dict:
        """
        运行所有对比测试
        
        Args:
            test_data: 包含所有测试所需数据的字典
            
        Returns:
            测试结果汇总
        """
        results = {
            'llm_prefill': None,
            'flow_encoder': None,
            'hift_generator': None,
            'streaming': None,
            'summary': {}
        }
        
        # LLM Prefill 测试
        if 'token_ids' in test_data:
            results['llm_prefill'] = self.compare_llm_prefill(
                test_data['token_ids'],
                test_data.get('prompt_tokens')
            )
        
        # Flow Encoder 测试
        if all(k in test_data for k in ['token_ids', 'prompt_tokens', 'prompt_mel', 'embedding']):
            results['flow_encoder'] = self.compare_flow_encoder(
                test_data['token_ids'],
                test_data['prompt_tokens'],
                test_data['prompt_mel'],
                test_data['embedding']
            )
        
        # HiFT Generator 测试
        if 'mel_spec' in test_data:
            results['hift_generator'] = self.compare_hift_generator(
                test_data['mel_spec']
            )
        
        # 汇总结果
        total_tests = sum(1 for v in results.values() if v is not None and isinstance(v, dict))
        passed_tests = sum(1 for v in results.values() 
                          if v is not None and isinstance(v, dict) and v.get('passed', False))
        
        results['summary'] = {
            'total_tests': total_tests,
            'passed_tests': passed_tests,
            'failed_tests': total_tests - passed_tests,
            'all_passed': passed_tests == total_tests
        }
        
        return results


def load_cpp_outputs_from_file(file_path: str) -> Dict[str, np.ndarray]:
    """
    从二进制文件加载 C++ 输出
    
    文件格式约定：
    - 每个张量：{name_length}{name}{dtype}{shape}{data}
    """
    outputs = {}
    
    with open(file_path, 'rb') as f:
        while True:
            # 读取名称长度
            name_len_bytes = f.read(4)
            if len(name_len_bytes) < 4:
                break
                
            name_len = int.from_bytes(name_len_bytes, 'little')
            name = f.read(name_len).decode('utf-8')
            
            # 读取数据类型
            dtype_bytes = f.read(4)
            dtype_code = int.from_bytes(dtype_bytes, 'little')
            dtype_map = {0: np.float32, 1: np.float16, 2: np.int32}
            dtype = dtype_map.get(dtype_code, np.float32)
            
            # 读取形状
            shape_len = int.from_bytes(f.read(4), 'little')
            shape = tuple(int.from_bytes(f.read(4), 'little') for _ in range(shape_len))
            
            # 读取数据
            size = np.prod(shape)
            data = np.frombuffer(f.read(size * np.dtype(dtype).itemsize), dtype=dtype)
            data = data.reshape(shape)
            
            outputs[name] = data
    
    return outputs


def main():
    parser = argparse.ArgumentParser(description='CosyVoice2 C++ vs Python 一致性测试')
    parser.add_argument('--model-dir', type=str, required=True, help='模型目录')
    parser.add_argument('--config', type=str, required=True, help='Python 模型配置文件')
    parser.add_argument('--cpp-outputs', type=str, help='C++ 输出二进制文件')
    parser.add_argument('--device', type=str, default='cpu', choices=['cpu', 'cuda'])
    parser.add_argument('--test-type', type=str, default='all', 
                       choices=['llm', 'flow', 'hift', 'streaming', 'all'])
    parser.add_argument('--tolerance', type=float, default=1e-3, help='误差容忍度')
    
    args = parser.parse_args()
    
    # 初始化对比器
    comparator = CosyVoice2Comparator(args.model_dir, args.device)
    
    # 加载 Python 模型
    try:
        comparator.load_python_model(args.config)
    except Exception as e:
        print(f"✗ Failed to load Python model: {e}")
        sys.exit(1)
    
    # 加载 C++ 输出
    if args.cpp_outputs and os.path.exists(args.cpp_outputs):
        cpp_outputs = load_cpp_outputs_from_file(args.cpp_outputs)
        comparator.set_cpp_output(cpp_outputs)
    
    # 设置容忍度
    comparator.tolerance['flow_mel'] = args.tolerance
    
    # 生成测试数据（实际使用时应从文件或其他来源加载）
    test_data = generate_test_data(comparator.python_model, args.device)
    
    # 运行测试
    print("\n" + "="*60)
    print("Running CosyVoice2 Consistency Tests")
    print("="*60 + "\n")
    
    results = comparator.run_all_tests(test_data)
    
    # 打印结果
    print_test_results(results)
    
    # 返回退出码
    sys.exit(0 if results['summary']['all_passed'] else 1)


def generate_test_data(python_model, device) -> Dict:
    """生成随机测试数据"""
    # 实际应用中应该使用真实的测试数据
    test_data = {
        'token_ids': np.random.randint(0, 6561, size=(1, 100), dtype=np.int32),
        'prompt_tokens': np.random.randint(0, 6561, size=(1, 50), dtype=np.int32),
        'prompt_mel': np.random.randn(1, 100, 80).astype(np.float32),
        'embedding': np.random.randn(192).astype(np.float32),
        'mel_spec': np.random.randn(1, 80, 200).astype(np.float32)
    }
    return test_data


def print_test_results(results: Dict):
    """打印测试结果"""
    for test_name, result in results.items():
        if test_name == 'summary':
            continue
            
        if result is None:
            print(f"⊘ {test_name}: Skipped")
            continue
            
        status = "✓ PASSED" if result.get('passed', False) else "✗ FAILED"
        print(f"{status} | {test_name}")
        
        if 'max_abs_error' in result:
            print(f"  Max Abs Error: {result['max_abs_error']:.6e}")
        if 'max_rel_error' in result:
            print(f"  Max Rel Error: {result['max_rel_error']:.6e}")
        if 'snr_db' in result:
            print(f"  SNR: {result['snr_db']:.2f} dB")
        if 'details' in result:
            print(f"  Details: {result['details']}")
        print()
    
    # 汇总
    summary = results['summary']
    print("="*60)
    print(f"Summary: {summary['passed_tests']}/{summary['total_tests']} tests passed")
    if summary['all_passed']:
        print("✓ All tests PASSED!")
    else:
        print(f"✗ {summary['failed_tests']} test(s) FAILED")
    print("="*60)


if __name__ == '__main__':
    main()
