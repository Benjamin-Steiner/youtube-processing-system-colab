"""
Memory Manager for GPU optimization
Implements intelligent memory management for multiple Faster-Whisper models
"""

import time
import logging
import threading
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from enum import Enum

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    logging.warning("PyTorch not available. GPU memory management will be limited.")


class MemoryStrategy(Enum):
    """Memory management strategies"""
    STATIC = "static"
    DYNAMIC = "dynamic"
    HYBRID = "hybrid"


@dataclass
class MemoryConfig:
    """Configuration for memory management"""
    strategy: MemoryStrategy = MemoryStrategy.HYBRID
    max_gpu_memory_gb: float = 16.0
    whisper_memory_per_instance_gb: float = 4.5
    audio_memory_per_file_gb: float = 0.5
    safety_margin_gb: float = 2.0
    cleanup_threshold: float = 0.85  # 85% memory usage


class ModelCacheManager:
    """
    LRU cache for Whisper models with intelligent memory management
    """
    
    def __init__(self, max_models: int = 3):
        self.max_models = max_models
        self.loaded_models = {}
        self.model_memory_usage = {}
        self.access_patterns = {}
        self.lock = threading.Lock()
        self.logger = logging.getLogger(__name__)
    
    def load_model(self, model_name: str, compute_type: str = 'float16') -> Optional[Any]:
        """Load a Whisper model with memory management"""
        with self.lock:
            # Check if model is already loaded
            if model_name in self.loaded_models:
                self.access_patterns[model_name] = time.time()
                self.logger.info(f"Model {model_name} already loaded, returning cached version")
                return self.loaded_models[model_name]
            
            # Check if we need to evict a model
            if len(self.loaded_models) >= self.max_models:
                self._evict_least_used_model()
            
            # Load the model
            try:
                from faster_whisper import WhisperModel
                
                model = WhisperModel(
                    model_name,
                    device="cuda",
                    compute_type=compute_type,
                    cpu_threads=4,
                    num_workers=1
                )
                
                # Estimate memory usage
                estimated_memory = self._estimate_model_memory(model_name, compute_type)
                
                self.loaded_models[model_name] = model
                self.model_memory_usage[model_name] = estimated_memory
                self.access_patterns[model_name] = time.time()
                
                self.logger.info(f"Loaded model {model_name} (Estimated memory: {estimated_memory:.1f}GB)")
                return model
                
            except Exception as e:
                self.logger.error(f"Failed to load model {model_name}: {e}")
                return None
    
    def _evict_least_used_model(self):
        """Evict the least recently used model"""
        if not self.loaded_models:
            return
        
        # Find least recently used model
        lru_model = min(self.access_patterns.keys(), 
                       key=lambda x: self.access_patterns[x])
        
        # Remove from memory
        del self.loaded_models[lru_model]
        del self.model_memory_usage[lru_model]
        del self.access_patterns[lru_model]
        
        # Force garbage collection
        if TORCH_AVAILABLE:
            torch.cuda.empty_cache()
        
        self.logger.info(f"Evicted model {lru_model} from cache")
    
    def _estimate_model_memory(self, model_name: str, compute_type: str) -> float:
        """Estimate memory usage for a model"""
        # Base memory estimates for different model sizes
        base_memory = {
            'tiny': 1.0,
            'base': 1.5,
            'small': 2.5,
            'medium': 4.0,
            'large': 6.0,
            'large-v2': 6.0,
            'large-v3': 6.0
        }
        
        # Get base memory for model
        base = base_memory.get(model_name, 4.5)  # Default to large-v2
        
        # Apply compute type factor
        compute_factors = {
            'float32': 1.0,
            'float16': 0.5,
            'int8': 0.25
        }
        
        factor = compute_factors.get(compute_type, 0.5)
        estimated_memory = base * factor
        
        return estimated_memory
    
    def get_cache_status(self) -> Dict[str, Any]:
        """Get status of the model cache"""
        return {
            'loaded_models': list(self.loaded_models.keys()),
            'model_count': len(self.loaded_models),
            'max_models': self.max_models,
            'memory_usage': self.model_memory_usage.copy(),
            'access_patterns': self.access_patterns.copy()
        }


class AudioMemoryManager:
    """
    Memory manager for audio files with compression and cleanup
    """
    
    def __init__(self, max_audio_memory_gb: float = 8.0):
        self.max_audio_memory_gb = max_audio_memory_gb
        self.active_audio_files = {}
        self.audio_memory_usage = {}
        self.compression_levels = {
            'high': 0.3,    # 30% of original size
            'medium': 0.5,  # 50% of original size
            'low': 0.8      # 80% of original size
        }
        self.lock = threading.Lock()
        self.logger = logging.getLogger(__name__)
    
    def load_audio_file(self, file_path: str, priority: str = 'normal') -> Optional[Any]:
        """Load an audio file with memory management"""
        with self.lock:
            # Estimate memory usage
            estimated_memory = self._estimate_audio_memory(file_path)
            
            # Check if we can load it
            if not self._can_load_audio(estimated_memory):
                self._optimize_audio_memory()
            
            # Load with appropriate compression
            compression = self._get_compression_level(priority)
            try:
                audio_data = self._load_with_compression(file_path, compression)
                
                self.active_audio_files[file_path] = audio_data
                self.audio_memory_usage[file_path] = estimated_memory
                
                self.logger.info(f"Loaded audio file {file_path} (Memory: {estimated_memory:.1f}GB)")
                return audio_data
                
            except Exception as e:
                self.logger.error(f"Failed to load audio file {file_path}: {e}")
                return None
    
    def _estimate_audio_memory(self, file_path: str) -> float:
        """Estimate memory usage for an audio file"""
        import os
        
        file_size = os.path.getsize(file_path)
        # Rough estimate: 1MB file ≈ 0.1GB in memory
        estimated_memory = file_size / (1024 * 1024 * 10)
        return estimated_memory
    
    def _can_load_audio(self, estimated_memory: float) -> bool:
        """Check if we can load an audio file"""
        current_memory = sum(self.audio_memory_usage.values())
        return (current_memory + estimated_memory) <= self.max_audio_memory_gb
    
    def _get_compression_level(self, priority: str) -> float:
        """Get compression level based on priority"""
        return self.compression_levels.get(priority, 0.5)
    
    def _load_with_compression(self, file_path: str, compression_factor: float) -> Any:
        """Load audio file with compression"""
        import librosa
        
        # Load audio with reduced sample rate for compression
        target_sr = int(16000 * compression_factor)  # Base sample rate * compression factor
        
        audio, sr = librosa.load(file_path, sr=target_sr)
        return {'audio': audio, 'sample_rate': sr, 'file_path': file_path}
    
    def _optimize_audio_memory(self):
        """Optimize audio memory usage"""
        # Remove low-priority audio files first
        low_priority_files = [
            f for f, data in self.active_audio_files.items()
            if self._get_priority(f) == 'low'
        ]
        
        for file_path in low_priority_files:
            self._remove_audio_file(file_path)
        
        # Compress remaining files if needed
        if self._get_total_audio_memory() > self.max_audio_memory_gb * 0.9:
            self._compress_audio_files()
    
    def _get_priority(self, file_path: str) -> str:
        """Get priority for a file (simplified implementation)"""
        # In a real implementation, this would be based on file metadata or user settings
        return 'normal'
    
    def _remove_audio_file(self, file_path: str):
        """Remove an audio file from memory"""
        if file_path in self.active_audio_files:
            del self.active_audio_files[file_path]
            del self.audio_memory_usage[file_path]
            self.logger.info(f"Removed audio file {file_path} from memory")
    
    def _get_total_audio_memory(self) -> float:
        """Get total audio memory usage"""
        return sum(self.audio_memory_usage.values())
    
    def _compress_audio_files(self):
        """Compress audio files to reduce memory usage"""
        for file_path, data in self.active_audio_files.items():
            if data['sample_rate'] > 8000:  # Only compress if not already compressed
                # Recompress with lower quality
                compressed_data = self._load_with_compression(file_path, 0.3)
                self.active_audio_files[file_path] = compressed_data
                self.logger.info(f"Compressed audio file {file_path}")


class HybridMemoryManager:
    """
    Hybrid intelligent memory manager combining model and audio management
    """
    
    def __init__(self, config: MemoryConfig):
        self.config = config
        self.model_cache = ModelCacheManager(max_models=3)
        self.audio_manager = AudioMemoryManager(max_audio_memory_gb=8.0)
        self.memory_predictor = MemoryPredictor()
        self.optimization_scheduler = OptimizationScheduler()
        self.lock = threading.Lock()
        self.logger = logging.getLogger(__name__)
    
    def get_concurrent_limit(self) -> int:
        """Calculate optimal concurrent limit based on available memory"""
        if self.config.strategy == MemoryStrategy.STATIC:
            return self._calculate_static_limit()
        else:
            return self._calculate_dynamic_limit()
    
    def _calculate_static_limit(self) -> int:
        """Calculate static concurrent limit"""
        usable_memory = self.config.max_gpu_memory_gb - self.config.safety_margin_gb
        whisper_instances = int(usable_memory / self.config.whisper_memory_per_instance_gb)
        audio_files = int(usable_memory / self.config.audio_memory_per_file_gb)
        
        return min(whisper_instances, audio_files, 3)  # Cap at 3 for safety
    
    def _calculate_dynamic_limit(self) -> int:
        """Calculate dynamic concurrent limit based on current usage"""
        current_usage = self._get_current_memory_usage()
        available_memory = self.config.max_gpu_memory_gb - current_usage - self.config.safety_margin_gb
        
        if available_memory <= 0:
            return 1  # Minimum concurrency
        
        # Calculate how many instances we can run
        whisper_instances = int(available_memory / self.config.whisper_memory_per_instance_gb)
        audio_files = int(available_memory / self.config.audio_memory_per_file_gb)
        
        return min(whisper_instances, audio_files, 5)  # Cap at 5 for safety
    
    def _get_current_memory_usage(self) -> float:
        """Get current GPU memory usage"""
        if TORCH_AVAILABLE:
            try:
                return torch.cuda.memory_allocated() / (1024**3)  # Convert to GB
            except:
                return 0.0
        return 0.0
    
    def optimize_memory_allocation(self):
        """Optimize memory allocation based on strategy"""
        if self.config.strategy == MemoryStrategy.AUTO:
            self._dynamic_optimization()
        else:
            self._static_optimization()
        
        # Predictive optimization
        if self.memory_predictor.predict_oom_risk():
            self._preemptive_cleanup()
    
    def _dynamic_optimization(self):
        """Dynamic memory optimization"""
        current_usage = self._get_current_memory_usage()
        usage_percentage = current_usage / self.config.max_gpu_memory_gb
        
        if usage_percentage > self.config.cleanup_threshold:
            self.logger.warning(f"High memory usage: {usage_percentage:.1%}")
            self._cleanup_memory()
    
    def _static_optimization(self):
        """Static memory optimization"""
        # For static strategy, just ensure we're within limits
        if self._get_current_memory_usage() > self.config.max_gpu_memory_gb * 0.9:
            self._cleanup_memory()
    
    def _cleanup_memory(self):
        """Clean up memory"""
        if TORCH_AVAILABLE:
            torch.cuda.empty_cache()
        
        # Clear some audio files
        self.audio_manager._optimize_audio_memory()
        
        self.logger.info("Memory cleanup completed")
    
    def _preemptive_cleanup(self):
        """Preemptive memory cleanup before OOM"""
        self.logger.info("Performing preemptive memory cleanup")
        self._cleanup_memory()
    
    def get_memory_status(self) -> Dict[str, Any]:
        """Get current memory status"""
        return {
            'total_memory_gb': self.config.max_gpu_memory_gb,
            'current_usage_gb': self._get_current_memory_usage(),
            'available_memory_gb': self.config.max_gpu_memory_gb - self._get_current_memory_usage(),
            'usage_percentage': self._get_current_memory_usage() / self.config.max_gpu_memory_gb,
            'model_cache_status': self.model_cache.get_cache_status(),
            'audio_memory_usage': self.audio_manager._get_total_audio_memory()
        }


class MemoryPredictor:
    """Predictive memory management"""
    
    def __init__(self):
        self.prediction_history = []
        self.logger = logging.getLogger(__name__)
    
    def predict_oom_risk(self) -> bool:
        """Predict if OOM risk is high"""
        # Simplified prediction - in real implementation would use ML
        return False


class OptimizationScheduler:
    """Scheduler for memory optimization tasks"""
    
    def __init__(self):
        self.scheduled_tasks = []
        self.logger = logging.getLogger(__name__)
    
    def schedule_optimization(self, task: callable, delay: float):
        """Schedule a memory optimization task"""
        # Simplified implementation
        pass 