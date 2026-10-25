import json
import time
import torch
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional

class TrainingLogger:

    def __init__(self, log_dir: str, model_name: str, config: Dict[str, Any], gpu_info: Optional[Dict[str, Any]]=None, resume: bool=False):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.model_name = model_name
        self.log_file = self.log_dir / f"{model_name.lower().replace(' ', '_')}_fair.json"
        if resume and self.log_file.exists():
            self._load_checkpoint()
        else:
            self._initialize_log(config, gpu_info)

    def _initialize_log(self, config: Dict[str, Any], gpu_info: Optional[Dict[str, Any]]):
        self.log_data = {'metadata': {'model_name': self.model_name, 'start_time': datetime.now().isoformat(), 'end_time': None, 'duration_hours': None, 'config': config, 'gpu_info': gpu_info or self._get_gpu_info()}, 'model_info': {'total_params': None, 'trainable_params': None, 'trainable_percentage': None}, 'training': {'steps': [], 'eval_steps': [], 'best_eval_loss': float('inf'), 'best_step': None, 'final_train_loss': None}, 'resources': {'peak_vram_gb': 0.0, 'avg_vram_gb': 0.0, 'gpu_utilization_percent': None}}

    def _load_checkpoint(self):
        with open(self.log_file, 'r') as f:
            self.log_data = json.load(f)
        print(f'Resumed from checkpoint: {self.log_file}')

    def _get_gpu_info(self) -> Dict[str, Any]:
        if not torch.cuda.is_available():
            return {'available': False}
        return {'available': True, 'name': torch.cuda.get_device_name(0), 'vram_gb': torch.cuda.get_device_properties(0).total_memory / 1000000000.0, 'device_count': torch.cuda.device_count()}

    def log_model_info(self, total_params: int, trainable_params: int):
        self.log_data['model_info']['total_params'] = total_params
        self.log_data['model_info']['trainable_params'] = trainable_params
        self.log_data['model_info']['trainable_percentage'] = 100 * trainable_params / total_params if total_params > 0 else 0

    def log_training_step(self, step: int, epoch: float, loss: float, lr: float):
        self.log_data['training']['steps'].append({'step': step, 'epoch': round(epoch, 4), 'loss': round(loss, 4), 'learning_rate': lr, 'timestamp': datetime.now().isoformat()})
        self.log_data['training']['final_train_loss'] = loss

    def log_eval_step(self, step: int, epoch: float, eval_loss: float, token_accuracy: Optional[float]=None):
        eval_entry = {'step': step, 'epoch': round(epoch, 4), 'eval_loss': round(eval_loss, 4), 'timestamp': datetime.now().isoformat()}
        if token_accuracy is not None:
            eval_entry['token_accuracy'] = round(token_accuracy, 4)
        self.log_data['training']['eval_steps'].append(eval_entry)
        if eval_loss < self.log_data['training']['best_eval_loss']:
            self.log_data['training']['best_eval_loss'] = eval_loss
            self.log_data['training']['best_step'] = step

    def log_resource_usage(self, peak_vram_gb: float, avg_vram_gb: Optional[float]=None):
        self.log_data['resources']['peak_vram_gb'] = round(peak_vram_gb, 2)
        if avg_vram_gb is not None:
            self.log_data['resources']['avg_vram_gb'] = round(avg_vram_gb, 2)

    def update_vram_tracking(self):
        if torch.cuda.is_available():
            current_vram = torch.cuda.max_memory_allocated() / 1000000000.0
            if current_vram > self.log_data['resources']['peak_vram_gb']:
                self.log_data['resources']['peak_vram_gb'] = round(current_vram, 2)

    def finalize(self):
        self.log_data['metadata']['end_time'] = datetime.now().isoformat()
        start_time = datetime.fromisoformat(self.log_data['metadata']['start_time'])
        end_time = datetime.fromisoformat(self.log_data['metadata']['end_time'])
        duration_seconds = (end_time - start_time).total_seconds()
        self.log_data['metadata']['duration_hours'] = round(duration_seconds / 3600, 2)

    def save(self):
        with open(self.log_file, 'w') as f:
            json.dump(self.log_data, f, indent=2)

    def print_summary(self):
        print('\n' + '=' * 60)
        print('Training Summary')
        print('=' * 60)
        print(f'Model: {self.model_name}')
        print(f"Duration: {self.log_data['metadata']['duration_hours']:.2f} hours")
        print(f"Peak VRAM: {self.log_data['resources']['peak_vram_gb']:.2f} GB")
        print(f"Final train loss: {self.log_data['training']['final_train_loss']:.4f}")
        print(f"Best eval loss: {self.log_data['training']['best_eval_loss']:.4f}")
        print(f"Best step: {self.log_data['training']['best_step']}")
        if self.log_data['model_info']['total_params']:
            print(f"Total params: {self.log_data['model_info']['total_params']:,}")
            print(f"Trainable params: {self.log_data['model_info']['trainable_params']:,}")
            print(f"Trainable %: {self.log_data['model_info']['trainable_percentage']:.2f}%")
        print(f'Log saved to: {self.log_file}')
        print('=' * 60)
