"""
Hybrid Adaptive Concurrency Manager
Implements manual and auto modes for video processing concurrency
"""

import time
import threading
from typing import Optional, Dict, List, Any
from dataclasses import dataclass
from enum import Enum
import logging

from ..utils.resource_monitor import ResourceMonitor
from ..optimization.memory_manager import HybridMemoryManager, MemoryConfig


class ProcessingMode(Enum):
    """Processing modes for concurrency management"""
    MANUAL = "manual"
    AUTO = "auto"
    HYBRID = "hybrid"


@dataclass
class ConcurrencyConfig:
    """Configuration for concurrency management"""
    mode: ProcessingMode = ProcessingMode.AUTO
    manual_concurrent: int = 3
    min_concurrent: int = 1
    max_concurrent: int = 5
    adjustment_interval: float = 5.0  # seconds
    memory_threshold: float = 0.85  # 85% GPU memory usage


class HybridConcurrencyManager:
    """
    Hybrid Adaptive Concurrency Manager
    Supports manual mode with user-specified concurrency and auto mode with dynamic adjustment
    """
    
    def __init__(self, config: ConcurrencyConfig):
        self.config = config
        self.current_concurrent = config.manual_concurrent
        self.active_processes = 0
        self.queue = []
        self.processing_history = []
        
        # Initialize components
        self.resource_monitor = ResourceMonitor()
        self.memory_manager = HybridMemoryManager(MemoryConfig())
        
        # Threading for background monitoring
        self.monitoring_thread = None
        self.should_monitor = False
        self.lock = threading.Lock()
        
        # Performance tracking
        self.performance_metrics = {
            'total_processed': 0,
            'successful_processing': 0,
            'failed_processing': 0,
            'average_processing_time': 0.0
        }
        
        self.logger = logging.getLogger(__name__)
    
    def start_monitoring(self):
        """Start background monitoring for auto mode"""
        if self.config.mode == ProcessingMode.AUTO:
            self.should_monitor = True
            self.monitoring_thread = threading.Thread(target=self._monitor_resources)
            self.monitoring_thread.daemon = True
            self.monitoring_thread.start()
            self.logger.info("Started resource monitoring for auto mode")
    
    def stop_monitoring(self):
        """Stop background monitoring"""
        self.should_monitor = False
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=5.0)
        self.logger.info("Stopped resource monitoring")
    
    def get_concurrent_limit(self) -> int:
        """Get the current concurrent processing limit"""
        if self.config.mode == ProcessingMode.MANUAL:
            return self.config.manual_concurrent
        else:
            return self.current_concurrent
    
    def can_start_process(self) -> bool:
        """Check if a new process can be started"""
        with self.lock:
            return self.active_processes < self.get_concurrent_limit()
    
    def start_process(self, video_url: str) -> bool:
        """Start a new processing process"""
        with self.lock:
            if self.can_start_process():
                self.active_processes += 1
                self.logger.info(f"Started processing: {video_url} (Active: {self.active_processes})")
                return True
            else:
                self.logger.warning(f"Cannot start process: {video_url} (Active: {self.active_processes}, Limit: {self.get_concurrent_limit()})")
                return False
    
    def finish_process(self, video_url: str, success: bool = True, processing_time: float = 0.0):
        """Mark a process as finished"""
        with self.lock:
            self.active_processes = max(0, self.active_processes - 1)
            
            # Update performance metrics
            self.performance_metrics['total_processed'] += 1
            if success:
                self.performance_metrics['successful_processing'] += 1
            else:
                self.performance_metrics['failed_processing'] += 1
            
            # Update average processing time
            if processing_time > 0:
                current_avg = self.performance_metrics['average_processing_time']
                total_processed = self.performance_metrics['total_processed']
                self.performance_metrics['average_processing_time'] = (
                    (current_avg * (total_processed - 1) + processing_time) / total_processed
                )
            
            self.logger.info(f"Finished processing: {video_url} (Active: {self.active_processes})")
    
    def switch_mode(self, new_mode: ProcessingMode, manual_concurrent: Optional[int] = None):
        """Switch between processing modes"""
        old_mode = self.config.mode
        self.config.mode = new_mode
        
        if manual_concurrent is not None:
            self.config.manual_concurrent = manual_concurrent
        
        # Handle mode-specific actions
        if new_mode == ProcessingMode.AUTO and old_mode != ProcessingMode.AUTO:
            self.start_monitoring()
        elif old_mode == ProcessingMode.AUTO and new_mode != ProcessingMode.AUTO:
            self.stop_monitoring()
        
        self.logger.info(f"Switched from {old_mode.value} to {new_mode.value} mode")
    
    def _monitor_resources(self):
        """Background monitoring for auto mode"""
        while self.should_monitor:
            try:
                self._adjust_concurrency()
                time.sleep(self.config.adjustment_interval)
            except Exception as e:
                self.logger.error(f"Error in resource monitoring: {e}")
    
    def _adjust_concurrency(self):
        """Adjust concurrency based on resource availability"""
        gpu_memory = self.resource_monitor.get_gpu_memory_usage()
        cpu_usage = self.resource_monitor.get_cpu_usage()
        
        # Calculate optimal concurrency
        optimal_concurrent = self._calculate_optimal_concurrency(gpu_memory, cpu_usage)
        
        # Apply adjustment with smoothing
        if abs(optimal_concurrent - self.current_concurrent) >= 1:
            old_concurrent = self.current_concurrent
            self.current_concurrent = optimal_concurrent
            
            self.logger.info(
                f"Adjusted concurrency: {old_concurrent} -> {self.current_concurrent} "
                f"(GPU: {gpu_memory:.1f}%, CPU: {cpu_usage:.1f}%)"
            )
    
    def _calculate_optimal_concurrency(self, gpu_memory: float, cpu_usage: float) -> int:
        """Calculate optimal concurrency based on resource usage"""
        # Base calculation on GPU memory
        if gpu_memory > 90:  # Critical memory usage
            base_concurrent = max(1, self.current_concurrent - 1)
        elif gpu_memory > 80:  # High memory usage
            base_concurrent = max(1, self.current_concurrent - 1)
        elif gpu_memory < 60 and cpu_usage < 70:  # Low resource usage
            base_concurrent = min(self.config.max_concurrent, self.current_concurrent + 1)
        else:  # Stable conditions
            base_concurrent = self.current_concurrent
        
        # Apply safety constraints
        optimal_concurrent = max(
            self.config.min_concurrent,
            min(self.config.max_concurrent, base_concurrent)
        )
        
        return optimal_concurrent
    
    def get_status(self) -> Dict[str, Any]:
        """Get current status of the concurrency manager"""
        return {
            'mode': self.config.mode.value,
            'current_concurrent': self.current_concurrent,
            'active_processes': self.active_processes,
            'queue_length': len(self.queue),
            'performance_metrics': self.performance_metrics.copy(),
            'resource_usage': {
                'gpu_memory': self.resource_monitor.get_gpu_memory_usage(),
                'cpu_usage': self.resource_monitor.get_cpu_usage()
            }
        }
    
    def add_to_queue(self, video_url: str, priority: str = 'normal'):
        """Add a video to the processing queue"""
        queue_item = {
            'url': video_url,
            'priority': priority,
            'added_time': time.time()
        }
        self.queue.append(queue_item)
        self.logger.info(f"Added to queue: {video_url} (Priority: {priority})")
    
    def get_next_from_queue(self) -> Optional[Dict[str, Any]]:
        """Get the next video from the queue based on priority"""
        if not self.queue:
            return None
        
        # Sort by priority (high, normal, low)
        priority_order = {'high': 0, 'normal': 1, 'low': 2}
        self.queue.sort(key=lambda x: priority_order.get(x['priority'], 1))
        
        return self.queue.pop(0)
    
    def get_queue_status(self) -> Dict[str, Any]:
        """Get status of the processing queue"""
        priority_counts = {}
        for item in self.queue:
            priority = item['priority']
            priority_counts[priority] = priority_counts.get(priority, 0) + 1
        
        return {
            'total_items': len(self.queue),
            'priority_breakdown': priority_counts,
            'oldest_item_age': time.time() - min([item['added_time'] for item in self.queue]) if self.queue else 0
        } 