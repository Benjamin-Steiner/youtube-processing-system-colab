#!/usr/bin/env python3
"""
YouTube Video Processing System - Main Entry Point
Downloads YouTube videos and converts them to YouTube Shorts with transcription
"""

import sys
import logging
import argparse
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / 'src'))

from src.models.video_processor import VideoProcessor, ProcessingConfig
from src.ui.colab_interface import ColabInterface


def setup_logging(level: str = 'INFO'):
    """Setup logging configuration"""
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('youtube_processing.log')
        ]
    )


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description='YouTube Video Processing System',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process a single video
  python main.py --video "https://www.youtube.com/watch?v=VIDEO_ID"
  
  # Process multiple videos with custom settings
  python main.py --videos video_list.txt --model large-v2 --concurrent 3
  
  # Run in Google Colab mode
  python main.py --colab --video "https://www.youtube.com/watch?v=VIDEO_ID"
        """
    )
    
    # Video input options
    parser.add_argument('--video', type=str, help='Single video URL to process')
    parser.add_argument('--videos', type=str, help='File containing list of video URLs')
    parser.add_argument('--urls', nargs='+', help='Multiple video URLs to process')
    
    # Processing options
    parser.add_argument('--model', type=str, default='large-v2',
                       choices=['tiny', 'base', 'small', 'medium', 'large', 'large-v2', 'large-v3'],
                       help='Whisper model to use for transcription')
    parser.add_argument('--compute-type', type=str, default='float16',
                       choices=['float32', 'float16', 'int8'],
                       help='Compute type for Whisper model')
    parser.add_argument('--concurrent', type=int, default=3,
                       help='Number of concurrent processing threads')
    parser.add_argument('--output-dir', type=str, default='./downloads',
                       help='Output directory for processed files')
    
    # Storage options
    parser.add_argument('--save-to-drive', action='store_true', default=True,
                       help='Save processed files to Google Drive')
    parser.add_argument('--drive-folder', type=str, default='YouTube_Processing',
                       help='Google Drive folder name')
    
    # Mode options
    parser.add_argument('--colab', action='store_true',
                       help='Run in Google Colab mode with UI')
    parser.add_argument('--headless', action='store_true',
                       help='Run in headless mode without UI')
    
    # Logging options
    parser.add_argument('--log-level', type=str, default='INFO',
                       choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                       help='Logging level')
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.log_level)
    logger = logging.getLogger(__name__)
    
    # Validate input
    video_urls = []
    if args.video:
        video_urls.append(args.video)
    elif args.videos:
        try:
            with open(args.videos, 'r') as f:
                video_urls = [line.strip() for line in f if line.strip()]
        except FileNotFoundError:
            logger.error(f"Video list file not found: {args.videos}")
            return 1
    elif args.urls:
        video_urls = args.urls
    else:
        logger.error("No video URLs provided. Use --video, --videos, or --urls")
        return 1
    
    if not video_urls:
        logger.error("No valid video URLs found")
        return 1
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Configure processing
    config = ProcessingConfig(
        whisper_model=args.model,
        compute_type=args.compute_type,
        max_concurrent=args.concurrent,
        output_dir=str(output_dir),
        save_to_drive=args.save_to_drive,
        drive_folder=args.drive_folder
    )
    
    try:
        # Initialize processor
        processor = VideoProcessor(config)
        
        # Add videos to queue
        for url in video_urls:
            processor.add_video(url)
        
        logger.info(f"Added {len(video_urls)} videos to processing queue")
        
        # Run in appropriate mode
        if args.colab:
            # Run with Colab interface
            interface = ColabInterface(processor)
            interface.run()
        else:
            # Run in headless mode
            logger.info("Starting processing in headless mode...")
            processor.start_processing()
            
            # Monitor progress
            try:
                while True:
                    status = processor.get_status()
                    logger.info(f"Status: {status['completed_count']} completed, "
                              f"{status['failed_count']} failed, "
                              f"{status['queue_length']} in queue")
                    
                    if status['queue_length'] == 0 and status['active_processes'] == 0:
                        break
                    
                    import time
                    time.sleep(5)
                    
            except KeyboardInterrupt:
                logger.info("Stopping processing...")
                processor.stop_processing()
            
            # Show results
            results = processor.get_results()
            logger.info(f"Processing complete!")
            logger.info(f"Successfully processed: {len(results['completed'])}")
            logger.info(f"Failed: {len(results['failed'])}")
            
            if results['completed']:
                logger.info("Completed videos:")
                for result in results['completed']:
                    logger.info(f"  - {result.video_title} ({result.processing_time:.1f}s)")
            
            if results['failed']:
                logger.warning("Failed videos:")
                for result in results['failed']:
                    logger.warning(f"  - {result.video_url}: {result.error_message}")
        
        return 0
        
    except Exception as e:
        logger.error(f"Error during processing: {e}")
        return 1


if __name__ == '__main__':
    sys.exit(main()) 