import json
import time
import torch
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional

class EvaluationLogger:

    def __init__(self, log_dir: str, model_name: str, model_path: str, adapter_path: Optional[str]=None, packed: bool=False, few_shot: int=0, gpu_info: Optional[Dict[str, Any]]=None):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.model_name = model_name
        self.few_shot = few_shot
        if adapter_path and few_shot > 0:
            suffix = f'ft_fewshot{few_shot}'
        elif few_shot > 0:
            suffix = f'fewshot{few_shot}'
        elif adapter_path:
            suffix = 'ft'
        else:
            suffix = 'zero'
        self.log_file = self.log_dir / f"{model_name.lower().replace(' ', '_')}_{suffix}.json"
        self.log_data = {'metadata': {'model_name': model_name, 'model_path': model_path, 'adapter_path': adapter_path, 'packed': packed, 'start_time': datetime.now().isoformat(), 'end_time': None, 'duration_minutes': None, 'gpu_info': gpu_info or self._get_gpu_info()}, 'model_info': {'total_params': None, 'model_size_mb': None, 'quantization': 'ternary-packed' if packed else 'bf16'}, 'efficiency': {'peak_vram_gb': None, 'latency_ms': None, 'perplexity': None}, 'benchmarks': {}}
        self.current_benchmark = None
        self.benchmark_start_time = None

    def _get_gpu_info(self) -> Dict[str, Any]:
        if not torch.cuda.is_available():
            return {'available': False}
        return {'available': True, 'name': torch.cuda.get_device_name(0), 'vram_gb': torch.cuda.get_device_properties(0).total_memory / 1000000000.0}

    def log_model_info(self, total_params: int, model_size_mb: float):
        self.log_data['model_info']['total_params'] = total_params
        self.log_data['model_info']['model_size_mb'] = round(model_size_mb, 1)

    def log_efficiency(self, peak_vram_gb: float, latency_ms: float, perplexity: Optional[float]=None):
        self.log_data['efficiency']['peak_vram_gb'] = round(peak_vram_gb, 2)
        self.log_data['efficiency']['latency_ms'] = round(latency_ms, 1)
        if perplexity is not None:
            self.log_data['efficiency']['perplexity'] = round(perplexity, 2)

    def start_benchmark(self, benchmark_name: str):
        self.current_benchmark = benchmark_name
        self.benchmark_start_time = time.time()
        print(f'\n--- {benchmark_name} ---')

    def end_benchmark(self, benchmark_name: str, accuracy: float, samples: int):
        if self.current_benchmark != benchmark_name:
            print(f'Warning: Ending benchmark {benchmark_name} but started {self.current_benchmark}')
        duration = time.time() - self.benchmark_start_time
        self.log_data['benchmarks'][benchmark_name] = {'accuracy': round(accuracy, 4), 'samples': samples, 'duration_minutes': round(duration / 60, 2)}
        print(f'  Accuracy: {accuracy:.1%}')
        print(f'  Duration: {duration / 60:.2f} minutes')
        self.current_benchmark = None
        self.benchmark_start_time = None

    def finalize(self):
        self.log_data['metadata']['end_time'] = datetime.now().isoformat()
        start_time = datetime.fromisoformat(self.log_data['metadata']['start_time'])
        end_time = datetime.fromisoformat(self.log_data['metadata']['end_time'])
        duration_seconds = (end_time - start_time).total_seconds()
        self.log_data['metadata']['duration_minutes'] = round(duration_seconds / 60, 2)

    def save(self):
        with open(self.log_file, 'w') as f:
            json.dump(self.log_data, f, indent=2)

    def print_summary(self):
        print('\n' + '=' * 60)
        print('Evaluation Summary')
        print('=' * 60)
        print(f'Model: {self.model_name}')
        print(f"Duration: {self.log_data['metadata']['duration_minutes']:.2f} minutes")
        if self.log_data['model_info']['model_size_mb']:
            print(f"Model size: {self.log_data['model_info']['model_size_mb']:.1f} MB")
        if self.log_data['efficiency']['peak_vram_gb']:
            print(f"Peak VRAM: {self.log_data['efficiency']['peak_vram_gb']:.2f} GB")
            print(f"Latency: {self.log_data['efficiency']['latency_ms']:.1f} ms")
        print('\nBenchmarks:')
        for (bench, data) in self.log_data['benchmarks'].items():
            print(f"  {bench}: {data['accuracy']:.1%} ({data['duration_minutes']:.2f} min)")
        print(f'\nLog saved to: {self.log_file}')
        print('=' * 60)
