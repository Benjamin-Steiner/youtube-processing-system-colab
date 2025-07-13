"""
Resource Monitor for GPU and CPU monitoring
Provides real-time resource usage tracking for the concurrency manager
"""

import time
import psutil
import logging
from typing import Dict, Any, Optional
from dataclasses import dataclass

try:
    import GPUtil
    GPUTIL_AVAILABLE = True
except ImportError:
    GPUTIL_AVAILABLE = False
    logging.warning("GPUtil not available. GPU monitoring will be disabled.")


@dataclass
class ResourceMetrics:
    """Resource usage metrics"""
    gpu_memory_usage: float = 0.0
    gpu_memory_total: float = 0.0
    gpu_memory_available: float = 0.0
    cpu_usage: float = 0.0
    system_memory_usage: float = 0.0
    system_memory_total: float = 0.0
    timestamp: float = 0.0


class ResourceMonitor:
    """
    Real-time resource monitoring for GPU and CPU
    Provides metrics for concurrency management decisions
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.metrics_history = []
        self.max_history_size = 100
        
        # Initialize GPU monitoring
        self.gpu_available = GPUTIL_AVAILABLE
        if self.gpu_available:
            try:
                self.gpus = GPUtil.getGPUs()
                self.logger.info(f"GPU monitoring initialized. Found {len(self.gpus)} GPU(s)")
            except Exception as e:
                self.logger.error(f"Failed to initialize GPU monitoring: {e}")
                self.gpu_available = False
        else:
            self.logger.warning("GPU monitoring disabled - GPUtil not available")
    
    def get_current_metrics(self) -> ResourceMetrics:
        """Get current resource usage metrics"""
        timestamp = time.time()
        
        # Get CPU usage
        cpu_usage = psutil.cpu_percent(interval=0.1)
        
        # Get system memory
        memory = psutil.virtual_memory()
        system_memory_usage = memory.percent
        system_memory_total = memory.total / (1024**3)  # Convert to GB
        
        # Get GPU metrics
        gpu_memory_usage = 0.0
        gpu_memory_total = 0.0
        gpu_memory_available = 0.0
        
        if self.gpu_available:
            try:
                gpu_metrics = self._get_gpu_metrics()
                gpu_memory_usage = gpu_metrics.get('memory_usage', 0.0)
                gpu_memory_total = gpu_metrics.get('memory_total', 0.0)
                gpu_memory_available = gpu_metrics.get('memory_available', 0.0)
            except Exception as e:
                self.logger.error(f"Error getting GPU metrics: {e}")
        
        metrics = ResourceMetrics(
            gpu_memory_usage=gpu_memory_usage,
            gpu_memory_total=gpu_memory_total,
            gpu_memory_available=gpu_memory_available,
            cpu_usage=cpu_usage,
            system_memory_usage=system_memory_usage,
            system_memory_total=system_memory_total,
            timestamp=timestamp
        )
        
        # Store in history
        self.metrics_history.append(metrics)
        if len(self.metrics_history) > self.max_history_size:
            self.metrics_history.pop(0)
        
        return metrics
    
    def get_gpu_memory_usage(self) -> float:
        """Get current GPU memory usage percentage"""
        if not self.gpu_available:
            return 0.0
        
        try:
            gpu_metrics = self._get_gpu_metrics()
            return gpu_metrics.get('memory_usage', 0.0)
        except Exception as e:
            self.logger.error(f"Error getting GPU memory usage: {e}")
            return 0.0
    
    def get_cpu_usage(self) -> float:
        """Get current CPU usage percentage"""
        return psutil.cpu_percent(interval=0.1)
    
    def get_system_memory_usage(self) -> float:
        """Get current system memory usage percentage"""
        memory = psutil.virtual_memory()
        return memory.percent
    
    def _get_gpu_metrics(self) -> Dict[str, float]:
        """Get detailed GPU metrics"""
        if not self.gpu_available or not self.gpus:
            return {'memory_usage': 0.0, 'memory_total': 0.0, 'memory_available': 0.0}
        
        # Get the first GPU (T4 in our case)
        gpu = self.gpus[0]
        
        memory_usage = gpu.memoryUtil * 100  # Convert to percentage
        memory_total = gpu.memoryTotal / 1024  # Convert to GB
        memory_available = (gpu.memoryTotal - gpu.memoryUsed) / 1024  # Convert to GB
        
        return {
            'memory_usage': memory_usage,
            'memory_total': memory_total,
            'memory_available': memory_available
        }
    
    def get_resource_summary(self) -> Dict[str, Any]:
        """Get a summary of current resource usage"""
        metrics = self.get_current_metrics()
        
        return {
            'gpu': {
                'memory_usage_percent': metrics.gpu_memory_usage,
                'memory_total_gb': metrics.gpu_memory_total,
                'memory_available_gb': metrics.gpu_memory_available,
                'available': self.gpu_available
            },
            'cpu': {
                'usage_percent': metrics.cpu_usage
            },
            'system_memory': {
                'usage_percent': metrics.system_memory_usage,
                'total_gb': metrics.system_memory_total
            },
            'timestamp': metrics.timestamp
        }
    
    def get_historical_metrics(self, duration_seconds: float = 300) -> list:
        """Get historical metrics for the specified duration"""
        current_time = time.time()
        cutoff_time = current_time - duration_seconds
        
        return [
            metrics for metrics in self.metrics_history
            if metrics.timestamp >= cutoff_time
        ]
    
    def get_average_metrics(self, duration_seconds: float = 60) -> ResourceMetrics:
        """Get average metrics over the specified duration"""
        historical = self.get_historical_metrics(duration_seconds)
        
        if not historical:
            return ResourceMetrics()
        
        # Calculate averages
        avg_gpu_memory = sum(m.gpu_memory_usage for m in historical) / len(historical)
        avg_cpu_usage = sum(m.cpu_usage for m in historical) / len(historical)
        avg_system_memory = sum(m.system_memory_usage for m in historical) / len(historical)
        
        return ResourceMetrics(
            gpu_memory_usage=avg_gpu_memory,
            cpu_usage=avg_cpu_usage,
            system_memory_usage=avg_system_memory,
            timestamp=time.time()
        )
    
    def is_resource_available(self, required_gpu_memory_gb: float = 4.0) -> bool:
        """Check if sufficient resources are available for processing"""
        metrics = self.get_current_metrics()
        
        # Check GPU memory availability
        if self.gpu_available and metrics.gpu_memory_available < required_gpu_memory_gb:
            self.logger.warning(
                f"Insufficient GPU memory: {metrics.gpu_memory_available:.1f}GB available, "
                f"{required_gpu_memory_gb}GB required"
            )
            return False
        
        # Check CPU usage
        if metrics.cpu_usage > 90:
            self.logger.warning(f"High CPU usage: {metrics.cpu_usage:.1f}%")
            return False
        
        # Check system memory
        if metrics.system_memory_usage > 90:
            self.logger.warning(f"High system memory usage: {metrics.system_memory_usage:.1f}%")
            return False
        
        return True
    
    def get_resource_warnings(self) -> list:
        """Get any resource usage warnings"""
        warnings = []
        metrics = self.get_current_metrics()
        
        if self.gpu_available:
            if metrics.gpu_memory_usage > 90:
                warnings.append(f"Critical GPU memory usage: {metrics.gpu_memory_usage:.1f}%")
            elif metrics.gpu_memory_usage > 80:
                warnings.append(f"High GPU memory usage: {metrics.gpu_memory_usage:.1f}%")
        
        if metrics.cpu_usage > 90:
            warnings.append(f"Critical CPU usage: {metrics.cpu_usage:.1f}%")
        elif metrics.cpu_usage > 80:
            warnings.append(f"High CPU usage: {metrics.cpu_usage:.1f}%")
        
        if metrics.system_memory_usage > 90:
            warnings.append(f"Critical system memory usage: {metrics.system_memory_usage:.1f}%")
        elif metrics.system_memory_usage > 80:
            warnings.append(f"High system memory usage: {metrics.system_memory_usage:.1f}%")
        
        return warnings
    
    def log_resource_status(self):
        """Log current resource status"""
        summary = self.get_resource_summary()
        warnings = self.get_resource_warnings()
        
        self.logger.info("Resource Status:")
        self.logger.info(f"  GPU Memory: {summary['gpu']['memory_usage_percent']:.1f}% "
                        f"({summary['gpu']['memory_available_gb']:.1f}GB available)")
        self.logger.info(f"  CPU Usage: {summary['cpu']['usage_percent']:.1f}%")
        self.logger.info(f"  System Memory: {summary['system_memory']['usage_percent']:.1f}%")
        
        if warnings:
            self.logger.warning("Resource Warnings:")
            for warning in warnings:
                self.logger.warning(f"  - {warning}") 