"""
Main Video Processor for YouTube Video Processing System
Integrates all components: downloading, transcription, and storage
"""

import time
import logging
import threading
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass
from pathlib import Path
import json

from src.core.concurrency_manager import HybridConcurrencyManager, ConcurrencyConfig, ProcessingMode
from src.utils.resource_monitor import ResourceMonitor
from src.optimization.memory_manager import HybridMemoryManager, MemoryConfig, MemoryStrategy
from src.handlers.error_handler import IntelligentErrorHandler


@dataclass
class ProcessingConfig:
    """Configuration for video processing"""
    # Download settings
    download_format: str = 'best[height<=1080]'
    audio_format: str = 'bestaudio'
    output_dir: str = './downloads'
    
    # Transcription settings
    whisper_model: str = 'distil-large-v3'
    compute_type: str = 'float16'
    language: Optional[str] = None
    task: str = 'transcribe'
    
    # Storage settings
    save_to_drive: bool = True
    drive_folder: str = 'YouTube_Processing'
    
    # Processing settings
    max_concurrent: int = 3
    enable_parallel: bool = True
    auto_cleanup: bool = True


@dataclass
class ProcessingResult:
    """Result of video processing"""
    video_url: str
    video_title: str
    download_path: str
    audio_path: str
    transcription_path: str
    srt_path: Optional[str] = None
    json_path: Optional[str] = None
    processing_time: float
    success: bool
    error_message: Optional[str] = None
    metadata: Dict[str, Any] = None


class VideoProcessor:
    """
    Main video processor integrating all system components
    """
    
    def __init__(self, config: ProcessingConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Initialize components
        self._initialize_components()
        
        # Processing state
        self.processing_queue = []
        self.active_processes = {}
        self.completed_results = []
        self.failed_results = []
        
        # Threading
        self.processing_threads = []
        self.should_stop = False
        self.lock = threading.Lock()
    
    def _initialize_components(self):
        """Initialize all system components"""
        # Concurrency management
        concurrency_config = ConcurrencyConfig(
            mode=ProcessingMode.AUTO,
            manual_concurrent=self.config.max_concurrent,
            min_concurrent=1,
            max_concurrent=5
        )
        self.concurrency_manager = HybridConcurrencyManager(concurrency_config)
        
        # Memory management
        memory_config = MemoryConfig(
            strategy=MemoryStrategy.HYBRID,
            max_gpu_memory_gb=16.0,
            whisper_memory_per_instance_gb=4.5,
            audio_memory_per_file_gb=0.5
        )
        self.memory_manager = HybridMemoryManager(memory_config)
        
        # Resource monitoring
        self.resource_monitor = ResourceMonitor()
        
        # Error handling
        self.error_handler = IntelligentErrorHandler()
        
        # Start monitoring
        self.concurrency_manager.start_monitoring()
        
        self.logger.info("All system components initialized successfully")
    
    def add_video(self, video_url: str, priority: str = 'normal') -> bool:
        """Add a video to the processing queue"""
        with self.lock:
            self.processing_queue.append({
                'url': video_url,
                'priority': priority,
                'added_time': time.time(),
                'status': 'queued'
            })
            
            self.logger.info(f"Added video to queue: {video_url} (Priority: {priority})")
            return True
    
    def start_processing(self):
        """Start the processing system"""
        self.should_stop = False
        
        # Start processing threads
        for i in range(self.config.max_concurrent):
            thread = threading.Thread(target=self._processing_worker, args=(i,))
            thread.daemon = True
            thread.start()
            self.processing_threads.append(thread)
        
        self.logger.info(f"Started processing with {self.config.max_concurrent} workers")
    
    def stop_processing(self):
        """Stop the processing system"""
        self.should_stop = True
        
        # Wait for threads to finish
        for thread in self.processing_threads:
            thread.join(timeout=10.0)
        
        self.processing_threads.clear()
        self.logger.info("Processing stopped")
    
    def _processing_worker(self, worker_id: int):
        """Worker thread for processing videos"""
        self.logger.info(f"Worker {worker_id} started")
        
        while not self.should_stop:
            # Get next video from queue
            video_item = self._get_next_video()
            if not video_item:
                time.sleep(1.0)
                continue
            
            # Start processing
            if self.concurrency_manager.start_process(video_item['url']):
                try:
                    self._process_video(video_item, worker_id)
                except Exception as e:
                    self.logger.error(f"Error processing video {video_item['url']}: {e}")
                    self._handle_processing_error(video_item, e)
                finally:
                    self.concurrency_manager.finish_process(video_item['url'])
        
        self.logger.info(f"Worker {worker_id} stopped")
    
    def _get_next_video(self) -> Optional[Dict[str, Any]]:
        """Get next video from queue"""
        with self.lock:
            if not self.processing_queue:
                return None
            
            # Sort by priority
            priority_order = {'high': 0, 'normal': 1, 'low': 2}
            self.processing_queue.sort(key=lambda x: priority_order.get(x['priority'], 1))
            
            return self.processing_queue.pop(0)
    
    def _process_video(self, video_item: Dict[str, Any], worker_id: int):
        """Process a single video"""
        video_url = video_item['url']
        start_time = time.time()
        
        self.logger.info(f"Worker {worker_id} processing: {video_url}")
        
        try:
            # Step 1: Download video
            download_result = self._download_video(video_url)
            if not download_result['success']:
                raise Exception(f"Download failed: {download_result['error']}")
            
            # Step 2: Extract audio
            audio_result = self._extract_audio(download_result['video_path'])
            if not audio_result['success']:
                raise Exception(f"Audio extraction failed: {audio_result['error']}")
            
            # Step 3: Transcribe audio
            transcription_result = self._transcribe_audio(audio_result['audio_path'])
            if not transcription_result['success']:
                raise Exception(f"Transcription failed: {transcription_result['error']}")
            
            # Step 4: Save to Google Drive (if enabled)
            if self.config.save_to_drive:
                drive_result = self._save_to_drive(
                    download_result['video_path'],
                    audio_result['audio_path'],
                    transcription_result['transcription_path'],
                    transcription_result.get('srt_path'),
                    transcription_result.get('json_path')
                )
            
            # Create result
            processing_time = time.time() - start_time
            result = ProcessingResult(
                video_url=video_url,
                video_title=download_result.get('title', 'Unknown'),
                download_path=download_result['video_path'],
                audio_path=audio_result['audio_path'],
                transcription_path=transcription_result['transcription_path'],
                srt_path=transcription_result.get('srt_path'),
                json_path=transcription_result.get('json_path'),
                processing_time=processing_time,
                success=True,
                metadata={
                    'worker_id': worker_id,
                    'priority': video_item['priority'],
                    'drive_saved': self.config.save_to_drive
                }
            )
            
            # Store result
            with self.lock:
                self.completed_results.append(result)
            
            self.logger.info(f"Successfully processed: {video_url} (Time: {processing_time:.1f}s)")
            
        except Exception as e:
            self._handle_processing_error(video_item, e)
    
    def _download_video(self, video_url: str) -> Dict[str, Any]:
        """Download video using yt-dlp"""
        try:
            import yt_dlp
            
            # Configure yt-dlp
            ydl_opts = {
                'format': self.config.download_format,
                'outtmpl': f'{self.config.output_dir}/%(title)s.%(ext)s',
                'quiet': True,
                'no_warnings': True
            }
            
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                # Get video info
                info = ydl.extract_info(video_url, download=False)
                video_title = info.get('title', 'Unknown')
                
                # Download video
                ydl.download([video_url])
                
                # Find downloaded file
                video_path = f"{self.config.output_dir}/{video_title}.mp4"
                
                return {
                    'success': True,
                    'video_path': video_path,
                    'title': video_title
                }
                
        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }
    
    def _extract_audio(self, video_path: str) -> Dict[str, Any]:
        """Extract audio from video"""
        try:
            from moviepy.editor import VideoFileClip
            
            # Generate audio path
            audio_path = video_path.replace('.mp4', '.wav')
            
            # Extract audio
            video = VideoFileClip(video_path)
            audio = video.audio
            audio.write_audiofile(audio_path, verbose=False, logger=None)
            
            # Clean up
            video.close()
            audio.close()
            
            return {
                'success': True,
                'audio_path': audio_path
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }
    
    def _transcribe_audio(self, audio_path: str) -> Dict[str, Any]:
        """Transcribe audio using WhisperX and generate SRT/JSON with alignment"""
        try:
            import torch # Import torch here for multithreaded context
            import whisperx
            import gc
            
            # Load Whisper model
            # Using smaller model for alignment is often sufficient and faster
            whisper_model_name = "base" # This can be configured if needed
            whisper_batch_size = 8 # This can be configured if needed
            
            self.logger.info(f"Loading WhisperX model: {whisper_model_name} on device: cuda, compute_type: {self.config.compute_type}")
            model = whisperx.load_model(whisper_model_name, "cuda", compute_type=self.config.compute_type)

            # Load audio
            audio = whisperx.load_audio(audio_path)

            # Transcribe audio
            self.logger.info("Transcribing audio with WhisperX...")
            result = model.transcribe(audio, batch_size=whisper_batch_size, language=self.config.language)
            
            # Clear whisper model from GPU memory
            del model
            gc.collect()
            torch.cuda.empty_cache()

            # Load alignment model and metadata
            self.logger.info(f"Loading alignment model for language: {result['language']}")
            align_model, metadata = whisperx.load_align_model(language_code=result['language'], device="cuda")

            # Align word timestamps
            self.logger.info("Aligning word timestamps...")
            aligned_result = whisperx.align(result["segments"], align_model, metadata, audio, "cuda", return_char_alignments=False)
            
            # Clear alignment model from GPU memory
            del align_model
            gc.collect()
            torch.cuda.empty_cache()
            
            base_name = Path(audio_path).stem
            output_dir = Path(self.config.output_dir)
            
            # Save transcription (text)
            transcription_path = output_dir / f'{base_name}.txt'
            with open(transcription_path, 'w', encoding='utf-8') as f:
                for segment in aligned_result["segments"]:
                    f.write(f"[{segment['start']:.2f}s -> {segment['end']:.2f}s] {segment['text']}\n")

            # Save SRT file
            srt_path = output_dir / f'{base_name}.srt'
            with open(srt_path, 'w', encoding='utf-8') as f:
                for i, segment in enumerate(aligned_result["segments"]):
                    start = str(time.strftime('%H:%M:%S', time.gmtime(segment['start']))) + f",{int((segment['start'] % 1) * 1000):03d}"
                    end = str(time.strftime('%H:%M:%S', time.gmtime(segment['end']))) + f",{int((segment['end'] % 1) * 1000):03d}"
                    f.write(f"{i + 1}\n")
                    f.write(f"{start} --> {end}\n")
                    f.write(f"{segment['text'].strip()}\n\n")
            
            # Save word-level JSON file
            json_path = output_dir / f'{base_name}_word_level.json'
            word_level_data = []
            for segment in aligned_result["segments"]:
                if 'words' in segment:
                    for word in segment['words']:
                        word_level_data.append({
                            'word': word['word'],
                            'start': word['start'],
                            'end': word['end'],
                            'score': word.get('score', 0.0) # score might not always be present
                        })
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(word_level_data, f, indent=4)
            
            return {
                'success': True,
                'transcription_path': str(transcription_path),
                'srt_path': str(srt_path),
                'json_path': str(json_path),
                'language': result['language'],
                'language_probability': result['language_probability']
            }
            
        except Exception as e:
            self.logger.error(f"Error during transcription for {audio_path}: {e}", exc_info=True)
            return {
                'success': False,
                'error': str(e),
                'transcription_path': None,
                'srt_path': None,
                'json_path': None
            }

    def _save_to_drive(self, video_path: str, audio_path: str, transcription_path: str, srt_path: Optional[str] = None, json_path: Optional[str] = None) -> Dict[str, Any]:
        """Save processed files to Google Drive"""
        try:
            from pydrive2.auth import GoogleAuth
            from pydrive2.drive import GoogleDrive
            import shutil
            
            # Authenticate and create drive object
            gauth = GoogleAuth()
            # Try to load saved client credentials
            gauth.LoadCredentialsFile("mycreds.txt")
            if gauth.credentials is None:
                # Authenticate if they're not there
                gauth.LocalWebserverAuth()
            elif gauth.access_token_expired:
                # Refresh them if expired
                gauth.Refresh()
            else:
                # Initialize the saved creds
                gauth.Authorize()
            
            gauth.SaveCredentialsFile("mycreds.txt")  # Save the current credentials to a file
            drive = GoogleDrive(gauth)
            
            # Get or create base folder
            folder_title = self.config.drive_folder
            file_list = drive.ListFile({'q': f"'root' in parents and title='{folder_title}' and mimeType='application/vnd.google-apps.folder' and trashed=false"}).GetList()
            
            if not file_list:
                folder = drive.CreateFile({'title': folder_title, 'mimeType': 'application/vnd.google-apps.folder'})
                folder.Upload()
                folder_id = folder['id']
                self.logger.info(f"Created Google Drive folder: {folder_title}")
            else:
                folder_id = file_list[0]['id']
                self.logger.info(f"Found Google Drive folder: {folder_title}")
            
            # Create subfolders for better organization
            video_folder_id = self._get_or_create_drive_folder(drive, folder_id, 'videos')
            audio_folder_id = self._get_or_create_drive_folder(drive, folder_id, 'audio')
            transcription_folder_id = self._get_or_create_drive_folder(drive, folder_id, 'transcriptions')

            # Upload video file
            video_file_name = Path(video_path).name
            video_file = drive.CreateFile({'title': video_file_name, 'parents': [{'id': video_folder_id}]})
            video_file.SetContentFile(video_path)
            video_file.Upload()
            self.logger.info(f"Uploaded video to Drive: {video_file_name}")
            
            # Upload audio file
            audio_file_name = Path(audio_path).name
            audio_file = drive.CreateFile({'title': audio_file_name, 'parents': [{'id': audio_folder_id}]})
            audio_file.SetContentFile(audio_path)
            audio_file.Upload()
            self.logger.info(f"Uploaded audio to Drive: {audio_file_name}")
            
            # Upload transcription file
            transcription_file_name = Path(transcription_path).name
            transcription_file = drive.CreateFile({'title': transcription_file_name, 'parents': [{'id': transcription_folder_id}]})
            transcription_file.SetContentFile(transcription_path)
            transcription_file.Upload()
            self.logger.info(f"Uploaded transcription to Drive: {transcription_file_name}")

            # Upload SRT file if generated
            if srt_path:
                srt_file_name = Path(srt_path).name
                srt_file = drive.CreateFile({'title': srt_file_name, 'parents': [{'id': transcription_folder_id}]})
                srt_file.SetContentFile(srt_path)
                srt_file.Upload()
                self.logger.info(f"Uploaded SRT to Drive: {srt_file_name}")

            # Upload JSON file if generated
            if json_path:
                json_file_name = Path(json_path).name
                json_file = drive.CreateFile({'title': json_file_name, 'parents': [{'id': transcription_folder_id}]})
                json_file.SetContentFile(json_path)
                json_file.Upload()
                self.logger.info(f"Uploaded JSON to Drive: {json_file_name}")
            
            # Clean up local files
            if self.config.auto_cleanup:
                self.logger.info(f"Cleaning up local files for {video_file_name}")
                Path(video_path).unlink(missing_ok=True)
                Path(audio_path).unlink(missing_ok=True)
                Path(transcription_path).unlink(missing_ok=True)
                if srt_path:
                    Path(srt_path).unlink(missing_ok=True)
                if json_path:
                    Path(json_path).unlink(missing_ok=True)

            return {'success': True}
            
        except Exception as e:
            self.logger.error(f"Error saving to Google Drive: {e}", exc_info=True)
            return {'success': False, 'error': str(e)}

    def _get_or_create_drive_folder(self, drive, parent_id: str, folder_name: str) -> str:
        """Helper to get or create a folder in Google Drive"""
        file_list = drive.ListFile({'q': f"'{parent_id}' in parents and title='{folder_name}' and mimeType='application/vnd.google-apps.folder' and trashed=false"}).GetList()
        if not file_list:
            folder = drive.CreateFile({'title': folder_name, 'mimeType': 'application/vnd.google-apps.folder', 'parents': [{'id': parent_id}]})
            folder.Upload()
            self.logger.info(f"Created Google Drive subfolder: {folder_name}")
            return folder['id']
        else:
            return file_list[0]['id']
    
    def _handle_processing_error(self, video_item: Dict[str, Any], error: Exception):
        """Handle processing errors"""
        # Use intelligent error handler
        context = {
            'video_url': video_item['url'],
            'priority': video_item['priority'],
            'worker_id': threading.current_thread().name
        }
        
        error_result = self.error_handler.handle_error(error, context)
        
        # Create failed result
        result = ProcessingResult(
            video_url=video_item['url'],
            video_title='Unknown',
            download_path='',
            audio_path='',
            transcription_path='',
            srt_path=None,
            json_path=None,
            processing_time=0.0,
            success=False,
            error_message=str(error),
            metadata={
                'error_info': error_result['error_info'],
                'recovery_success': error_result['recovery_success']
            }
        )
        
        with self.lock:
            self.failed_results.append(result)
    
    def get_status(self) -> Dict[str, Any]:
        """Get current processing status"""
        with self.lock:
            return {
                'queue_length': len(self.processing_queue),
                'active_processes': len(self.active_processes),
                'completed_count': len(self.completed_results),
                'failed_count': len(self.failed_results),
                'concurrency_status': self.concurrency_manager.get_status(),
                'memory_status': self.memory_manager.get_memory_status(),
                'resource_status': self.resource_monitor.get_resource_summary(),
                'error_statistics': self.error_handler.get_error_statistics()
            }
    
    def get_results(self) -> Dict[str, List[ProcessingResult]]:
        """Get processing results"""
        with self.lock:
            return {
                'completed': self.completed_results.copy(),
                'failed': self.failed_results.copy()
            }
    
    def clear_results(self):
        """Clear processing results"""
        with self.lock:
            self.completed_results.clear()
            self.failed_results.clear()
    
    def get_recommendations(self) -> List[str]:
        """Get system recommendations"""
        recommendations = []
        
        # Get error handler recommendations
        error_recommendations = self.error_handler.get_recovery_recommendations()
        recommendations.extend(error_recommendations)
        
        # Get resource recommendations
        resource_warnings = self.resource_monitor.get_resource_warnings()
        if resource_warnings:
            recommendations.extend(resource_warnings)
        
        return recommendations 