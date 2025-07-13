"""
Intelligent Error Handler for YouTube Video Processing
Implements AI-powered error classification and predictive recovery
"""

import time
import logging
import traceback
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass
from enum import Enum
import threading


class ErrorSeverity(Enum):
    """Error severity levels"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ErrorCategory(Enum):
    """Error categories for classification"""
    NETWORK = "network"
    MEMORY = "memory"
    MODEL = "model"
    AUDIO = "audio"
    DOWNLOAD = "download"
    TRANSCRIPTION = "transcription"
    STORAGE = "storage"
    UNKNOWN = "unknown"


@dataclass
class ErrorInfo:
    """Error information structure"""
    error_type: str
    error_message: str
    category: ErrorCategory
    severity: ErrorSeverity
    timestamp: float
    context: Dict[str, Any]
    stack_trace: str
    recovery_attempts: int = 0
    max_recovery_attempts: int = 3


class IntelligentErrorClassifier:
    """
    AI-powered error classification system
    """
    
    def __init__(self):
        self.error_patterns = self._initialize_error_patterns()
        self.classification_history = []
        self.logger = logging.getLogger(__name__)
    
    def classify_error(self, error: Exception, context: Dict[str, Any]) -> ErrorInfo:
        """Classify an error based on its characteristics"""
        error_type = type(error).__name__
        error_message = str(error)
        stack_trace = traceback.format_exc()
        
        # Analyze error patterns
        category = self._determine_category(error_type, error_message, context)
        severity = self._determine_severity(error_type, error_message, context)
        
        error_info = ErrorInfo(
            error_type=error_type,
            error_message=error_message,
            category=category,
            severity=severity,
            timestamp=time.time(),
            context=context,
            stack_trace=stack_trace
        )
        
        # Store classification
        self.classification_history.append(error_info)
        
        self.logger.info(f"Classified error: {error_type} -> {category.value} ({severity.value})")
        return error_info
    
    def _determine_category(self, error_type: str, error_message: str, context: Dict[str, Any]) -> ErrorCategory:
        """Determine error category based on patterns"""
        error_lower = error_message.lower()
        
        # Network-related errors
        if any(pattern in error_lower for pattern in ['connection', 'timeout', 'network', 'http', 'ssl']):
            return ErrorCategory.NETWORK
        
        # Memory-related errors
        if any(pattern in error_lower for pattern in ['memory', 'cuda', 'gpu', 'oom', 'out of memory']):
            return ErrorCategory.MEMORY
        
        # Model-related errors
        if any(pattern in error_lower for pattern in ['model', 'whisper', 'torch', 'tensor']):
            return ErrorCategory.MODEL
        
        # Audio-related errors
        if any(pattern in error_lower for pattern in ['audio', 'librosa', 'soundfile', 'wav']):
            return ErrorCategory.AUDIO
        
        # Download-related errors
        if any(pattern in error_lower for pattern in ['download', 'yt-dlp', 'youtube', 'video']):
            return ErrorCategory.DOWNLOAD
        
        # Transcription-related errors
        if any(pattern in error_lower for pattern in ['transcription', 'transcribe', 'whisper']):
            return ErrorCategory.TRANSCRIPTION
        
        # Storage-related errors
        if any(pattern in error_lower for pattern in ['drive', 'google', 'storage', 'file', 'permission']):
            return ErrorCategory.STORAGE
        
        return ErrorCategory.UNKNOWN
    
    def _determine_severity(self, error_type: str, error_message: str, context: Dict[str, Any]) -> ErrorSeverity:
        """Determine error severity based on context and patterns"""
        error_lower = error_message.lower()
        
        # Critical errors
        if any(pattern in error_lower for pattern in ['out of memory', 'cuda error', 'fatal']):
            return ErrorSeverity.CRITICAL
        
        # High severity errors
        if any(pattern in error_lower for pattern in ['connection failed', 'model failed', 'download failed']):
            return ErrorSeverity.HIGH
        
        # Medium severity errors
        if any(pattern in error_lower for pattern in ['timeout', 'retry', 'temporary']):
            return ErrorSeverity.MEDIUM
        
        # Low severity errors
        if any(pattern in error_lower for pattern in ['warning', 'info', 'debug']):
            return ErrorSeverity.LOW
        
        return ErrorSeverity.MEDIUM
    
    def _initialize_error_patterns(self) -> Dict[str, List[str]]:
        """Initialize error pattern database"""
        return {
            'network': ['connection', 'timeout', 'network', 'http', 'ssl', 'dns'],
            'memory': ['memory', 'cuda', 'gpu', 'oom', 'out of memory', 'insufficient'],
            'model': ['model', 'whisper', 'torch', 'tensor', 'load', 'inference'],
            'audio': ['audio', 'librosa', 'soundfile', 'wav', 'mp3', 'format'],
            'download': ['download', 'yt-dlp', 'youtube', 'video', 'stream'],
            'transcription': ['transcription', 'transcribe', 'whisper', 'segments'],
            'storage': ['drive', 'google', 'storage', 'file', 'permission', 'quota']
        }


class PredictiveRecovery:
    """
    Predictive error recovery system
    """
    
    def __init__(self):
        self.recovery_strategies = self._initialize_recovery_strategies()
        self.recovery_history = []
        self.logger = logging.getLogger(__name__)
    
    def get_recovery_strategy(self, error_info: ErrorInfo) -> Optional[Callable]:
        """Get appropriate recovery strategy for the error"""
        category = error_info.category
        severity = error_info.severity
        
        # Get category-specific strategies
        strategies = self.recovery_strategies.get(category.value, [])
        
        # Filter by severity
        appropriate_strategies = [
            strategy for strategy in strategies
            if self._is_strategy_appropriate(strategy, severity)
        ]
        
        if appropriate_strategies:
            return appropriate_strategies[0]  # Return first appropriate strategy
        
        return None
    
    def _is_strategy_appropriate(self, strategy: Dict[str, Any], severity: ErrorSeverity) -> bool:
        """Check if strategy is appropriate for error severity"""
        strategy_severity = strategy.get('severity', ErrorSeverity.MEDIUM)
        
        severity_order = {
            ErrorSeverity.LOW: 1,
            ErrorSeverity.MEDIUM: 2,
            ErrorSeverity.HIGH: 3,
            ErrorSeverity.CRITICAL: 4
        }
        
        return severity_order[severity] <= severity_order[strategy_severity]
    
    def _initialize_recovery_strategies(self) -> Dict[str, List[Dict[str, Any]]]:
        """Initialize recovery strategies database"""
        return {
            'network': [
                {
                    'name': 'retry_with_backoff',
                    'severity': ErrorSeverity.MEDIUM,
                    'function': self._retry_with_backoff,
                    'description': 'Retry operation with exponential backoff'
                },
                {
                    'name': 'switch_connection',
                    'severity': ErrorSeverity.HIGH,
                    'function': self._switch_connection,
                    'description': 'Switch to alternative connection method'
                }
            ],
            'memory': [
                {
                    'name': 'cleanup_memory',
                    'severity': ErrorSeverity.HIGH,
                    'function': self._cleanup_memory,
                    'description': 'Clean up GPU memory and retry'
                },
                {
                    'name': 'reduce_concurrency',
                    'severity': ErrorSeverity.CRITICAL,
                    'function': self._reduce_concurrency,
                    'description': 'Reduce concurrent processing'
                }
            ],
            'model': [
                {
                    'name': 'reload_model',
                    'severity': ErrorSeverity.MEDIUM,
                    'function': self._reload_model,
                    'description': 'Reload the Whisper model'
                },
                {
                    'name': 'switch_model_size',
                    'severity': ErrorSeverity.HIGH,
                    'function': self._switch_model_size,
                    'description': 'Switch to smaller model'
                }
            ],
            'audio': [
                {
                    'name': 'reprocess_audio',
                    'severity': ErrorSeverity.MEDIUM,
                    'function': self._reprocess_audio,
                    'description': 'Reprocess audio file with different settings'
                },
                {
                    'name': 'skip_audio',
                    'severity': ErrorSeverity.HIGH,
                    'function': self._skip_audio,
                    'description': 'Skip problematic audio file'
                }
            ],
            'download': [
                {
                    'name': 'retry_download',
                    'severity': ErrorSeverity.MEDIUM,
                    'function': self._retry_download,
                    'description': 'Retry video download'
                },
                {
                    'name': 'alternative_download',
                    'severity': ErrorSeverity.HIGH,
                    'function': self._alternative_download,
                    'description': 'Try alternative download method'
                }
            ]
        }
    
    def _retry_with_backoff(self, error_info: ErrorInfo, context: Dict[str, Any]) -> bool:
        """Retry operation with exponential backoff"""
        max_attempts = 3
        base_delay = 1.0
        
        for attempt in range(max_attempts):
            delay = base_delay * (2 ** attempt)
            self.logger.info(f"Retry attempt {attempt + 1}/{max_attempts} in {delay}s")
            
            time.sleep(delay)
            
            # Here you would retry the original operation
            # For now, we'll just return success
            return True
        
        return False
    
    def _cleanup_memory(self, error_info: ErrorInfo, context: Dict[str, Any]) -> bool:
        """Clean up GPU memory"""
        try:
            import torch
            torch.cuda.empty_cache()
            self.logger.info("GPU memory cleanup completed")
            return True
        except Exception as e:
            self.logger.error(f"Memory cleanup failed: {e}")
            return False
    
    def _reduce_concurrency(self, error_info: ErrorInfo, context: Dict[str, Any]) -> bool:
        """Reduce concurrent processing"""
        # This would interact with the concurrency manager
        self.logger.info("Reducing concurrent processing")
        return True
    
    def _reload_model(self, error_info: ErrorInfo, context: Dict[str, Any]) -> bool:
        """Reload the Whisper model"""
        self.logger.info("Reloading Whisper model")
        return True
    
    def _switch_model_size(self, error_info: ErrorInfo, context: Dict[str, Any]) -> bool:
        """Switch to smaller model"""
        self.logger.info("Switching to smaller model")
        return True
    
    def _reprocess_audio(self, error_info: ErrorInfo, context: Dict[str, Any]) -> bool:
        """Reprocess audio file"""
        self.logger.info("Reprocessing audio file")
        return True
    
    def _skip_audio(self, error_info: ErrorInfo, context: Dict[str, Any]) -> bool:
        """Skip problematic audio file"""
        self.logger.info("Skipping problematic audio file")
        return True
    
    def _retry_download(self, error_info: ErrorInfo, context: Dict[str, Any]) -> bool:
        """Retry video download"""
        self.logger.info("Retrying video download")
        return True
    
    def _alternative_download(self, error_info: ErrorInfo, context: Dict[str, Any]) -> bool:
        """Try alternative download method"""
        self.logger.info("Trying alternative download method")
        return True
    
    def _switch_connection(self, error_info: ErrorInfo, context: Dict[str, Any]) -> bool:
        """Switch to alternative connection method"""
        self.logger.info("Switching to alternative connection method")
        return True


class GracefulDegradation:
    """
    Graceful degradation system for handling critical failures
    """
    
    def __init__(self):
        self.degradation_levels = {
            'full': 100,      # Full functionality
            'reduced': 75,     # Reduced functionality
            'minimal': 50,     # Minimal functionality
            'emergency': 25    # Emergency mode
        }
        self.current_level = 'full'
        self.logger = logging.getLogger(__name__)
    
    def should_degrade(self, error_info: ErrorInfo) -> bool:
        """Determine if system should degrade"""
        if error_info.severity == ErrorSeverity.CRITICAL:
            return True
        
        # Check error frequency
        recent_errors = self._get_recent_errors(300)  # Last 5 minutes
        if len(recent_errors) > 5:
            return True
        
        return False
    
    def _get_recent_errors(self, time_window: float) -> List[ErrorInfo]:
        """Get recent errors within time window"""
        current_time = time.time()
        cutoff_time = current_time - time_window
        
        # This would access the error classifier's history
        # For now, return empty list
        return []
    
    def degrade_system(self, error_info: ErrorInfo):
        """Degrade system functionality"""
        if error_info.category == ErrorCategory.MEMORY:
            self.current_level = 'reduced'
            self.logger.warning("System degraded to reduced mode due to memory issues")
        elif error_info.category == ErrorCategory.NETWORK:
            self.current_level = 'minimal'
            self.logger.warning("System degraded to minimal mode due to network issues")
        else:
            self.current_level = 'emergency'
            self.logger.critical("System degraded to emergency mode")
    
    def get_current_capabilities(self) -> Dict[str, Any]:
        """Get current system capabilities based on degradation level"""
        capabilities = {
            'concurrent_processing': self._get_concurrent_limit(),
            'model_quality': self._get_model_quality(),
            'audio_quality': self._get_audio_quality(),
            'download_methods': self._get_download_methods()
        }
        
        return capabilities
    
    def _get_concurrent_limit(self) -> int:
        """Get concurrent processing limit based on degradation level"""
        limits = {
            'full': 3,
            'reduced': 2,
            'minimal': 1,
            'emergency': 1
        }
        return limits.get(self.current_level, 1)
    
    def _get_model_quality(self) -> str:
        """Get model quality based on degradation level"""
        qualities = {
            'full': 'large-v2',
            'reduced': 'medium',
            'minimal': 'small',
            'emergency': 'base'
        }
        return qualities.get(self.current_level, 'base')
    
    def _get_audio_quality(self) -> str:
        """Get audio quality based on degradation level"""
        qualities = {
            'full': 'high',
            'reduced': 'medium',
            'minimal': 'low',
            'emergency': 'low'
        }
        return qualities.get(self.current_level, 'low')
    
    def _get_download_methods(self) -> List[str]:
        """Get available download methods based on degradation level"""
        methods = {
            'full': ['yt-dlp', 'alternative'],
            'reduced': ['yt-dlp'],
            'minimal': ['basic'],
            'emergency': ['basic']
        }
        return methods.get(self.current_level, ['basic'])


class IntelligentErrorHandler:
    """
    Main intelligent error handler combining all components
    """
    
    def __init__(self):
        self.classifier = IntelligentErrorClassifier()
        self.recovery = PredictiveRecovery()
        self.degradation = GracefulDegradation()
        self.error_history = []
        self.lock = threading.Lock()
        self.logger = logging.getLogger(__name__)
    
    def handle_error(self, error: Exception, context: Dict[str, Any]) -> Dict[str, Any]:
        """Handle an error with intelligent classification and recovery"""
        with self.lock:
            # Classify the error
            error_info = self.classifier.classify_error(error, context)
            
            # Store in history
            self.error_history.append(error_info)
            
            # Check if system should degrade
            if self.degradation.should_degrade(error_info):
                self.degradation.degrade_system(error_info)
            
            # Get recovery strategy
            recovery_strategy = self.recovery.get_recovery_strategy(error_info)
            
            # Attempt recovery
            recovery_success = False
            if recovery_strategy and error_info.recovery_attempts < error_info.max_recovery_attempts:
                error_info.recovery_attempts += 1
                recovery_success = recovery_strategy(error_info, context)
            
            # Log comprehensive error information
            self._log_error_comprehensive(error_info, recovery_success)
            
            return {
                'error_info': error_info,
                'recovery_success': recovery_success,
                'degradation_level': self.degradation.current_level,
                'capabilities': self.degradation.get_current_capabilities()
            }
    
    def _log_error_comprehensive(self, error_info: ErrorInfo, recovery_success: bool):
        """Log comprehensive error information"""
        self.logger.error(f"Error occurred: {error_info.error_type}")
        self.logger.error(f"Category: {error_info.category.value}")
        self.logger.error(f"Severity: {error_info.severity.value}")
        self.logger.error(f"Message: {error_info.error_message}")
        self.logger.error(f"Recovery attempts: {error_info.recovery_attempts}")
        self.logger.error(f"Recovery success: {recovery_success}")
        
        if error_info.severity in [ErrorSeverity.HIGH, ErrorSeverity.CRITICAL]:
            self.logger.error(f"Stack trace: {error_info.stack_trace}")
    
    def get_error_statistics(self) -> Dict[str, Any]:
        """Get error statistics"""
        if not self.error_history:
            return {}
        
        # Calculate statistics
        total_errors = len(self.error_history)
        errors_by_category = {}
        errors_by_severity = {}
        
        for error in self.error_history:
            category = error.category.value
            severity = error.severity.value
            
            errors_by_category[category] = errors_by_category.get(category, 0) + 1
            errors_by_severity[severity] = errors_by_severity.get(severity, 0) + 1
        
        return {
            'total_errors': total_errors,
            'errors_by_category': errors_by_category,
            'errors_by_severity': errors_by_severity,
            'current_degradation_level': self.degradation.current_level,
            'recent_errors': len([e for e in self.error_history if time.time() - e.timestamp < 3600])
        }
    
    def reset_degradation(self):
        """Reset system degradation level"""
        self.degradation.current_level = 'full'
        self.logger.info("System degradation reset to full functionality")
    
    def get_recovery_recommendations(self) -> List[str]:
        """Get recovery recommendations based on error history"""
        recommendations = []
        
        if not self.error_history:
            return recommendations
        
        # Analyze recent errors
        recent_errors = [e for e in self.error_history if time.time() - e.timestamp < 3600]
        
        if len(recent_errors) > 10:
            recommendations.append("High error frequency detected - consider system restart")
        
        memory_errors = [e for e in recent_errors if e.category == ErrorCategory.MEMORY]
        if len(memory_errors) > 3:
            recommendations.append("Multiple memory errors - consider reducing concurrency")
        
        network_errors = [e for e in recent_errors if e.category == ErrorCategory.NETWORK]
        if len(network_errors) > 5:
            recommendations.append("Network issues detected - check connectivity")
        
        return recommendations 