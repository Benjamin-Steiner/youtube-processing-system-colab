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
    processing_time: float
    success: bool
    error_message: Optional[str] = None
    metadata: Dict[str, Any] = None
    srt_path: Optional[str] = None
    json_path: Optional[str] = None


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
        self.logger.debug(f"_processing_worker {worker_id}: Entering main loop.")
        
        while not self.should_stop:
            self.logger.debug(f"_processing_worker {worker_id}: Checking queue. Current queue length: {len(self.processing_queue)}")
            # Get next video from queue
            video_item = self._get_next_video()
            if not video_item:
                self.logger.debug(f"_processing_worker {worker_id}: No video in queue, sleeping.")
                time.sleep(1.0)
                continue
            
            self.logger.debug(f"_processing_worker {worker_id}: Retrieved video: {video_item.get('url')}")
            
            # Start processing
            if self.concurrency_manager.start_process(video_item['url']):
                self.logger.info(f"_processing_worker {worker_id}: Concurrency manager approved, starting process for {video_item['url']}")
                try:
                    self._process_video(video_item, worker_id)
                except Exception as e:
                    self.logger.error(f"Error processing video {video_item['url']}: {e}")
                    self._handle_processing_error(video_item, e)
                finally:
                    self.concurrency_manager.finish_process(video_item['url'])
                    self.logger.info(f"_processing_worker {worker_id}: Finished process for {video_item['url']}")
            else:
                self.logger.warning(f"_processing_worker {worker_id}: Concurrency manager denied starting process for {video_item['url']}. Re-queuing.")
                # If concurrency manager denies, re-add to queue (or handle based on strategy)
                with self.lock:
                    self.processing_queue.append(video_item)
                time.sleep(1.0) # Small delay before retrying
        
        self.logger.info(f"Worker {worker_id} stopped")
    
    def _get_next_video(self) -> Optional[Dict[str, Any]]:
        """Get next video from queue"""
        self.logger.debug("_get_next_video: Attempting to get next video.")
        with self.lock:
            if not self.processing_queue:
                self.logger.debug("_get_next_video: Queue is empty, returning None.")
                return None
            
            # Sort by priority
            priority_order = {'high': 0, 'normal': 1, 'low': 2}
            self.processing_queue.sort(key=lambda x: priority_order.get(x['priority'], 1))
            
            video_item = self.processing_queue.pop(0)
            self.logger.debug(f"_get_next_video: Retrieved video {video_item.get('url')}. New queue length: {len(self.processing_queue)}")
            return video_item
    
    def _process_video(self, video_item: Dict[str, Any], worker_id: int):
        """Process a single video"""
        video_url = video_item['url']
        start_time = time.time()
        
        self.logger.info(f"Worker {worker_id} processing: {video_url}")
        self.logger.debug(f"_process_video: Starting processing for {video_url}")
        
        try:
            # Step 1: Download video
            self.logger.debug(f"_process_video: Calling _download_video for {video_url}")
            download_result = self._download_video(video_url)
            if not download_result['success']:
                self.logger.error(f"_process_video: Download failed for {video_url}: {download_result['error']}")
                raise Exception(f"Download failed: {download_result['error']}")
            self.logger.debug(f"_process_video: Download successful for {video_url}. Path: {download_result['video_path']}")
            
            # Step 2: Extract audio
            self.logger.debug(f"_process_video: Calling _extract_audio for {download_result['video_path']}")
            audio_result = self._extract_audio(download_result['video_path'])
            if not audio_result['success']:
                self.logger.error(f"_process_video: Audio extraction failed for {download_result['video_path']}: {audio_result['error']}")
                raise Exception(f"Audio extraction failed: {audio_result['error']}")
            self.logger.debug(f"_process_video: Audio extraction successful. Path: {audio_result['audio_path']}")
            
            # Step 3: Transcribe audio
            self.logger.debug(f"_process_video: Calling _transcribe_audio for {audio_result['audio_path']}")
            transcription_result = self._transcribe_audio(audio_result['audio_path'])
            if not transcription_result['success']:
                self.logger.error(f"_process_video: Transcription failed for {audio_result['audio_path']}: {transcription_result['error']}")
                raise Exception(f"Transcription failed: {transcription_result['error']}")
            self.logger.debug(f"_process_video: Transcription successful. TXT: {transcription_result['transcription_path']}, SRT: {transcription_result.get('srt_path')}, JSON: {transcription_result.get('json_path')}")
            
            # Step 4: Save to Google Drive (if enabled)
            if self.config.save_to_drive:
                self.logger.debug(f"_process_video: Calling _save_to_drive for {video_url}")
                drive_result = self._save_to_drive(
                    download_result['video_path'],
                    audio_result['audio_path'],
                    transcription_result['transcription_path'],
                    transcription_result.get('srt_path'),
                    transcription_result.get('json_path')
                )
                if not drive_result['success']:
                    self.logger.error(f"_process_video: Saving to Drive failed for {video_url}: {drive_result['error']}")
                    # Do not raise exception here, allow partial success if only drive save fails
            self.logger.debug(f"_process_video: Save to Drive operation completed for {video_url}")
            
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
            self.logger.debug(f"_process_video: Finished processing for {video_url}")
            
        except Exception as e:
            self.logger.error(f"_process_video: Caught exception during processing for {video_url}: {e}", exc_info=True)
            self._handle_processing_error(video_item, e)
    
    def _download_video(self, video_url: str) -> Dict[str, Any]:
        """Download video using yt-dlp"""
        self.logger.debug(f"_download_video: Starting download for {video_url}")
        try:
            import yt_dlp
            
            # Configure yt-dlp
            ydl_opts = {
                'format': self.config.download_format,
                'outtmpl': f'{self.config.output_dir}/%(title)s.%(ext)s',
                'quiet': True,
                'no_warnings': True
            }
            self.logger.debug(f"_download_video: yt-dlp options: {ydl_opts}")
            
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                # Get video info
                self.logger.debug(f"_download_video: Extracting info for {video_url}")
                info = ydl.extract_info(video_url, download=False)
                video_title = info.get('title', 'Unknown')
                self.logger.debug(f"_download_video: Video title: {video_title}")
                
                # Download video
                self.logger.debug(f"_download_video: Initiating download for {video_url}")
                ydl.download([video_url])
                
                # Find downloaded file (yt-dlp might add id or other info)
                # We need to be robust in finding the actual downloaded file name
                downloaded_filename = ydl.prepare_filename(info)
                # Adjust for potential .mp4 extension added by yt-dlp if original was different
                video_path = Path(downloaded_filename).with_suffix('.mp4')
                if not video_path.exists(): # Fallback if .mp4 wasn't the final extension
                    # This is a bit tricky with yt-dlp as it can output various extensions
                    # For now, let's assume it's mp4 or the original ext
                    # A more robust solution might iterate through files in output_dir
                    self.logger.warning(f"_download_video: Expected video_path {video_path} not found. Trying info.get('_filename').")
                    video_path = Path(info.get('_filename', downloaded_filename))
                    if not video_path.exists():
                        # If still not found, search in output_dir
                        for f in Path(self.config.output_dir).iterdir():
                            if video_title in f.stem:
                                video_path = f
                                self.logger.debug(f"_download_video: Found video file by title search: {video_path}")
                                break
                
                if not video_path.exists():
                    raise FileNotFoundError(f"Downloaded video file not found at expected path: {video_path}")
                
                self.logger.debug(f"_download_video: Download complete. Video path: {video_path}")
                
                return {
                    'success': True,
                    'video_path': str(video_path),
                    'title': video_title
                }
                
        except Exception as e:
            self.logger.error(f"_download_video: Error downloading video {video_url}: {e}", exc_info=True)
            return {
                'success': False,
                'error': str(e),
                'video_path': None,
                'title': None
            }
    
    def _extract_audio(self, video_path: str) -> Dict[str, Any]:
        """Extract audio from video"""
        self.logger.debug(f"_extract_audio: Starting audio extraction for {video_path}")
        try:
            from moviepy.editor import VideoFileClip
            
            # Generate audio path
            audio_path = Path(video_path).with_suffix('.wav')
            self.logger.debug(f"_extract_audio: Target audio path: {audio_path}")
            
            # Extract audio
            video = VideoFileClip(video_path)
            audio = video.audio
            self.logger.debug(f"_extract_audio: Writing audio file to {audio_path}")
            audio.write_audiofile(str(audio_path), verbose=False, logger=None)
            
            # Clean up
            video.close()
            audio.close()
            self.logger.debug(f"_extract_audio: Audio extraction complete for {video_path}")
            
            return {
                'success': True,
                'audio_path': str(audio_path)
            }
            
        except Exception as e:
            self.logger.error(f"_extract_audio: Error extracting audio from {video_path}: {e}", exc_info=True)
            return {
                'success': False,
                'error': str(e),
                'audio_path': None
            }
    
    def _transcribe_audio(self, audio_path: str) -> Dict[str, Any]:
        """Transcribe audio using WhisperX and generate SRT/JSON with alignment"""
        self.logger.debug(f"_transcribe_audio: Starting transcription for {audio_path}")
        try:
            import torch # Import torch here for multithreaded context
            import whisperx
            import gc
            
            # Load Whisper model
            # Using smaller model for alignment is often sufficient and faster
            whisper_model_name = self.config.whisper_model # Use configured model size
            whisper_batch_size = 8 # This can be configured if needed
            
            self.logger.info(f"Loading WhisperX model: {whisper_model_name} on device: cuda, compute_type: {self.config.compute_type}")
            model = whisperx.load_model(whisper_model_name, "cuda", compute_type=self.config.compute_type)

            # Load audio
            self.logger.debug(f"_transcribe_audio: Loading audio with whisperx for {audio_path}")
            audio = whisperx.load_audio(audio_path)

            # Transcribe audio
            self.logger.info("Transcribing audio with WhisperX...")
            result = model.transcribe(audio, batch_size=whisper_batch_size, language=self.config.language)
            self.logger.debug(f"_transcribe_audio: Initial transcription complete. Detected language: {result.get('language')}")
            
            # Clear whisper model from GPU memory
            del model
            gc.collect()
            torch.cuda.empty_cache()
            self.logger.debug(f"_transcribe_audio: Cleared Whisper model from GPU memory.")

            # Load alignment model and metadata
            self.logger.info(f"Loading alignment model for language: {result['language']}")
            align_model, metadata = whisperx.load_align_model(language_code=result['language'], device="cuda")

            # Align word timestamps
            self.logger.info("Aligning word timestamps...")
            aligned_result = whisperx.align(result["segments"], align_model, metadata, audio, "cuda", return_char_alignments=False)
            self.logger.debug(f"_transcribe_audio: Word alignment complete.")
            
            # Clear alignment model from GPU memory
            del align_model
            gc.collect()
            torch.cuda.empty_cache()
            self.logger.debug(f"_transcribe_audio: Cleared alignment model from GPU memory.")
            
            base_name = Path(audio_path).stem
            output_dir = Path(self.config.output_dir)
            output_dir.mkdir(parents=True, exist_ok=True) # Ensure output directory exists

            # Save transcription (text)
            transcription_path = output_dir / f'{base_name}.txt'
            self.logger.debug(f"_transcribe_audio: Saving transcription to {transcription_path}")
            with open(transcription_path, 'w', encoding='utf-8') as f:
                for segment in aligned_result["segments"]:
                    f.write(f"[{segment['start']:.2f}s -> {segment['end']:.2f}s] {segment['text']}\n")

            # Save SRT file
            srt_path = output_dir / f'{base_name}.srt'
            self.logger.debug(f"_transcribe_audio: Saving SRT to {srt_path}")
            with open(srt_path, 'w', encoding='utf-8') as f:
                for i, segment in enumerate(aligned_result["segments"]):
                    start = str(time.strftime('%H:%M:%S', time.gmtime(segment['start']))) + f",{int((segment['start'] % 1) * 1000):03d}"
                    end = str(time.strftime('%H:%M:%S', time.gmtime(segment['end']))) + f",{int((segment['end'] % 1) * 1000):03d}"
                    f.write(f"{i + 1}\n")
                    f.write(f"{start} --> {end}\n")
                    f.write(f"{segment['text'].strip()}\n\n")
            
            # Save word-level JSON file
            json_path = output_dir / f'{base_name}_word_level.json'
            self.logger.debug(f"_transcribe_audio: Saving JSON to {json_path}")
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
            
            self.logger.debug(f"_transcribe_audio: Transcription process complete for {audio_path}")
            return {
                'success': True,
                'transcription_path': str(transcription_path),
                'srt_path': str(srt_path),
                'json_path': str(json_path),
                'language': result['language'],
                'language_probability': result['language_probability']
            }
            
        except Exception as e:
            self.logger.error(f"_transcribe_audio: Error during transcription for {audio_path}: {e}", exc_info=True)
            return {
                'success': False,
                'error': str(e),
                'transcription_path': None,
                'srt_path': None,
                'json_path': None
            }

    def _save_to_drive(self, video_path: str, audio_path: str, transcription_path: str, srt_path: Optional[str] = None, json_path: Optional[str] = None) -> Dict[str, Any]:
        """Save processed files to Google Drive"""
        self.logger.debug(f"_save_to_drive: Starting save to Drive for video: {video_path}")
        try:
            from pydrive2.auth import GoogleAuth
            from pydrive2.drive import GoogleDrive
            import shutil
            
            # Authenticate and create drive object
            gauth = GoogleAuth()
            # Try to load saved client credentials
            self.logger.debug("_save_to_drive: Loading Drive credentials...")
            gauth.LoadCredentialsFile("mycreds.txt")
            if gauth.credentials is None:
                # Authenticate if they're not there
                self.logger.info("_save_to_drive: Authenticating with LocalWebserverAuth...")
                gauth.LocalWebserverAuth()
            elif gauth.access_token_expired:
                # Refresh them if expired
                self.logger.info("_save_to_drive: Refreshing Drive credentials...")
                gauth.Refresh()
            else:
                # Initialize the saved creds
                self.logger.info("_save_to_drive: Authorizing with saved Drive credentials...")
                gauth.Authorize()
            
            gauth.SaveCredentialsFile("mycreds.txt")  # Save the current credentials to a file
            drive = GoogleDrive(gauth)
            self.logger.debug("_save_to_drive: Google Drive authentication complete.")
            
            # Get or create base folder
            folder_title = self.config.drive_folder
            self.logger.debug(f"_save_to_drive: Checking for Drive folder: {folder_title}")
            file_list = drive.ListFile({'q': f"'root' in parents and title='{folder_title}' and mimeType='application/vnd.google-apps.folder' and trashed=false"}).GetList()
            
            if not file_list:
                folder = drive.CreateFile({'title': folder_title, 'mimeType': 'application/vnd.google-apps.folder'})
                folder.Upload()
                folder_id = folder['id']
                self.logger.info(f"Created Google Drive folder: {folder_title} with ID: {folder_id}")
            else:
                folder_id = file_list[0]['id']
                self.logger.info(f"Found Google Drive folder: {folder_title} with ID: {folder_id}")
            
            # Create subfolders for better organization
            video_folder_id = self._get_or_create_drive_folder(drive, folder_id, 'videos')
            audio_folder_id = self._get_or_create_drive_folder(drive, folder_id, 'audio')
            transcription_folder_id = self._get_or_create_drive_folder(drive, folder_id, 'transcriptions')

            # Upload video file
            video_file_name = Path(video_path).name
            self.logger.debug(f"_save_to_drive: Uploading video file: {video_file_name} to {video_folder_id}")
            video_file = drive.CreateFile({'title': video_file_name, 'parents': [{'id': video_folder_id}]})
            video_file.SetContentFile(video_path)
            video_file.Upload()
            self.logger.info(f"Uploaded video to Drive: {video_file_name}")
            
            # Upload audio file
            audio_file_name = Path(audio_path).name
            self.logger.debug(f"_save_to_drive: Uploading audio file: {audio_file_name} to {audio_folder_id}")
            audio_file = drive.CreateFile({'title': audio_file_name, 'parents': [{'id': audio_folder_id}]})
            audio_file.SetContentFile(audio_path)
            audio_file.Upload()
            self.logger.info(f"Uploaded audio to Drive: {audio_file_name}")
            
            # Upload transcription file
            transcription_file_name = Path(transcription_path).name
            self.logger.debug(f"_save_to_drive: Uploading transcription file: {transcription_file_name} to {transcription_folder_id}")
            transcription_file = drive.CreateFile({'title': transcription_file_name, 'parents': [{'id': transcription_folder_id}]})
            transcription_file.SetContentFile(transcription_path)
            transcription_file.Upload()
            self.logger.info(f"Uploaded transcription to Drive: {transcription_file_name}")

            # Upload SRT file if generated
            if srt_path:
                srt_file_name = Path(srt_path).name
                self.logger.debug(f"_save_to_drive: Uploading SRT file: {srt_file_name} to {transcription_folder_id}")
                srt_file = drive.CreateFile({'title': srt_file_name, 'parents': [{'id': transcription_folder_id}]})
                srt_file.SetContentFile(srt_path)
                srt_file.Upload()
                self.logger.info(f"Uploaded SRT to Drive: {srt_file_name}")

            # Upload JSON file if generated
            if json_path:
                json_file_name = Path(json_path).name
                self.logger.debug(f"_save_to_drive: Uploading JSON file: {json_file_name} to {transcription_folder_id}")
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
                self.logger.debug(f"_save_to_drive: Local files cleaned up for {video_file_name}")

            return {'success': True}
            
        except Exception as e:
            self.logger.error(f"_save_to_drive: Error saving to Google Drive: {e}", exc_info=True)
            return {'success': False, 'error': str(e)}

    def _get_or_create_drive_folder(self, drive, parent_id: str, folder_name: str) -> str:
        """Helper to get or create a folder in Google Drive"""
        self.logger.debug(f"_get_or_create_drive_folder: Checking for folder '{folder_name}' under parent ID: {parent_id}")
        file_list = drive.ListFile({'q': f"'{parent_id}' in parents and title='{folder_name}' and mimeType='application/vnd.google-apps.folder' and trashed=false"}).GetList()
        if not file_list:
            folder = drive.CreateFile({'title': folder_name, 'mimeType': 'application/vnd.google-apps.folder', 'parents': [{'id': parent_id}]})
            folder.Upload()
            self.logger.info(f"Created Google Drive subfolder: {folder_name} with ID: {folder['id']}")
            return folder['id']
        else:
            self.logger.debug(f"_get_or_create_drive_folder: Found existing folder: {folder_name} with ID: {file_list[0]['id']}")
            return file_list[0]['id']
    
    def _handle_processing_error(self, video_item: Dict[str, Any], error: Exception):
        """Handle processing errors"""
        self.logger.error(f"_handle_processing_error: Handling error for video {video_item['url']}")
        # Use intelligent error handler
        context = {
            'video_url': video_item['url'],
            'priority': video_item['priority'],
            'worker_id': threading.current_thread().name
        }
        
        error_result = self.error_handler.handle_error(error, context)
        self.logger.debug(f"_handle_processing_error: Error handler result: {error_result}")
        
        # Create failed result
        result = ProcessingResult(
            video_url=video_item['url'],
            video_title='Unknown', # Title might not be available on failure
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
        self.logger.info(f"_handle_processing_error: Video {video_item['url']} marked as failed.")

    def get_status(self) -> Dict[str, Any]:
        """Get current processing status"""
        with self.lock:
            status = {
                'queue_length': len(self.processing_queue),
                'active_processes': len(self.active_processes),
                'completed_count': len(self.completed_results),
                'failed_count': len(self.failed_results),
                'concurrency_status': self.concurrency_manager.get_status(),
                'memory_status': self.memory_manager.get_memory_status(),
                'resource_status': self.resource_monitor.get_resource_summary(),
                'error_statistics': self.error_handler.get_error_statistics()
            }
            self.logger.debug(f"get_status: Current status: {status}")
            return status

    def get_results(self) -> Dict[str, List[ProcessingResult]]:
        """Get processing results"""
        with self.lock:
            results = {
                'completed': self.completed_results.copy(),
                'failed': self.failed_results.copy()
            }
            self.logger.debug(f"get_results: Returning results. Completed: {len(results['completed'])}, Failed: {len(results['failed'])}")
            return results
    
    def clear_results(self):
        """Clear processing results"""
        with self.lock:
            self.completed_results.clear()
            self.failed_results.clear()
            self.logger.info("Cleared all processing results.")
    
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
        self.logger.debug(f"get_recommendations: Current recommendations: {recommendations}")
        return recommendations 