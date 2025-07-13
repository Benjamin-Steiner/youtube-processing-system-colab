# YouTube Video Processing System

A comprehensive system for downloading YouTube videos, extracting audio, and generating transcriptions using Faster-Whisper Turbo model on T4 GPU in Google Colab.

## 🚀 Features

- **Parallel Processing**: Intelligent concurrency management with hybrid adaptive architecture
- **GPU Optimization**: Dynamic memory management for multiple Whisper model instances
- **Error Recovery**: AI-powered error classification and predictive recovery
- **Google Colab Integration**: Interactive UI for Google Colab environment
- **Google Drive Storage**: Automatic saving to Google Drive
- **Resource Monitoring**: Real-time GPU and CPU monitoring
- **Graceful Degradation**: System adapts to resource constraints

## 📋 Requirements

- Python 3.8+
- Google Colab with T4 GPU
- Google Drive access
- Internet connection

## 🛠️ Installation

### For Google Colab

1. **Clone the repository**:

```python
!git clone https://github.com/your-repo/youtube-processing-system.git
%cd youtube-processing-system
```

2. **Install dependencies**:

```python
!pip install -r requirements.txt
```

3. **Setup Google Drive**:

```python
from google.colab import drive
drive.mount('/content/drive')
```

### For Local Development

1. **Clone the repository**:

```bash
git clone https://github.com/your-repo/youtube-processing-system.git
cd youtube-processing-system
```

2. **Install dependencies**:

```bash
pip install -r requirements.txt
```

## 🎯 Usage

### Google Colab Interface

```python
from src.models.video_processor import VideoProcessor, ProcessingConfig
from src.ui.colab_interface import ColabInterface

# Initialize processor
config = ProcessingConfig(
    whisper_model='large-v2',
    compute_type='float16',
    max_concurrent=3
)
processor = VideoProcessor(config)

# Run interface
interface = ColabInterface(processor)
interface.run()
```

### Command Line Interface

```bash
# Process a single video
python main.py --video "https://www.youtube.com/watch?v=VIDEO_ID"

# Process multiple videos
python main.py --videos video_list.txt --model large-v2 --concurrent 3

# Run in Google Colab mode
python main.py --colab --video "https://www.youtube.com/watch?v=VIDEO_ID"
```

### Programmatic Usage

```python
from src.models.video_processor import VideoProcessor, ProcessingConfig

# Configure processing
config = ProcessingConfig(
    whisper_model='large-v2',
    compute_type='float16',
    max_concurrent=3,
    save_to_drive=True
)

# Initialize processor
processor = VideoProcessor(config)

# Add videos to queue
processor.add_video("https://www.youtube.com/watch?v=VIDEO_ID", priority='high')

# Start processing
processor.start_processing()

# Monitor status
while True:
    status = processor.get_status()
    print(f"Completed: {status['completed_count']}, Failed: {status['failed_count']}")

    if status['queue_length'] == 0 and status['active_processes'] == 0:
        break

    time.sleep(5)

# Get results
results = processor.get_results()
for result in results['completed']:
    print(f"✓ {result.video_title} ({result.processing_time:.1f}s)")
```

## 🏗️ Architecture

### Core Components

1. **Hybrid Concurrency Manager** (`src/core/concurrency_manager.py`)

   - Manual and auto modes for concurrency control
   - Dynamic resource-based adjustment
   - Priority queue management

2. **Resource Monitor** (`src/utils/resource_monitor.py`)

   - Real-time GPU and CPU monitoring
   - Historical metrics tracking
   - Resource availability prediction

3. **Memory Manager** (`src/optimization/memory_manager.py`)

   - LRU cache for Whisper models
   - Audio file compression
   - Predictive memory cleanup

4. **Error Handler** (`src/handlers/error_handler.py`)

   - AI-powered error classification
   - Predictive recovery strategies
   - Graceful degradation system

5. **Video Processor** (`src/models/video_processor.py`)
   - Main processing pipeline
   - Integration of all components
   - Result management

### Processing Pipeline

```
Video URL → Download → Audio Extraction → Transcription → Google Drive Storage
    ↓           ↓            ↓              ↓              ↓
Concurrency  Resource    Memory        Error         Results
Management   Monitoring  Management    Handling      Tracking
```

## ⚙️ Configuration

### ProcessingConfig Options

```python
config = ProcessingConfig(
    # Download settings
    download_format='best[height<=1080]',
    audio_format='bestaudio',
    output_dir='./downloads',

    # Transcription settings
    whisper_model='large-v2',  # tiny, base, small, medium, large, large-v2, large-v3
    compute_type='float16',    # float32, float16, int8
    language=None,             # Auto-detect if None
    task='transcribe',         # transcribe or translate

    # Storage settings
    save_to_drive=True,
    drive_folder='YouTube_Processing',

    # Processing settings
    max_concurrent=3,
    enable_parallel=True,
    auto_cleanup=True
)
```

### Concurrency Modes

- **Manual Mode**: Fixed concurrency level
- **Auto Mode**: Dynamic adjustment based on resources
- **Hybrid Mode**: Combines manual and auto features

## 📊 Monitoring

### Status Information

```python
status = processor.get_status()

# Processing status
print(f"Queue: {status['queue_length']}")
print(f"Active: {status['active_processes']}")
print(f"Completed: {status['completed_count']}")
print(f"Failed: {status['failed_count']}")

# Resource status
resource = status['resource_status']
print(f"GPU Memory: {resource['gpu']['memory_usage_percent']:.1f}%")
print(f"CPU Usage: {resource['cpu']['usage_percent']:.1f}%")

# Concurrency status
concurrency = status['concurrency_status']
print(f"Mode: {concurrency['mode']}")
print(f"Current: {concurrency['current_concurrent']}")
```

### Error Statistics

```python
stats = processor.error_handler.get_error_statistics()
print(f"Total Errors: {stats['total_errors']}")
print(f"By Category: {stats['errors_by_category']}")
print(f"By Severity: {stats['errors_by_severity']}")
```

## 🔧 Troubleshooting

### Common Issues

1. **GPU Memory Errors**

   - Reduce concurrent processing
   - Use smaller Whisper model
   - Enable auto cleanup

2. **Download Failures**

   - Check internet connection
   - Verify video URL
   - Try alternative download method

3. **Transcription Errors**
   - Check audio file quality
   - Verify model loading
   - Check GPU availability

### System Recommendations

```python
recommendations = processor.get_recommendations()
for rec in recommendations:
    print(f"• {rec}")
```

## 📁 File Structure

```
youtube_processing_system/
├── src/
│   ├── core/
│   │   └── concurrency_manager.py
│   ├── utils/
│   │   └── resource_monitor.py
│   ├── optimization/
│   │   └── memory_manager.py
│   ├── handlers/
│   │   └── error_handler.py
│   ├── models/
│   │   └── video_processor.py
│   └── ui/
│       └── colab_interface.py
├── requirements.txt
├── main.py
└── README.md
```

## 🧪 Testing

```bash
# Run tests
python -m pytest tests/

# Run with coverage
python -m pytest --cov=src tests/
```

## 📈 Performance

### Benchmarks (T4 GPU)

| Model    | Memory Usage | Processing Time | Quality   |
| -------- | ------------ | --------------- | --------- |
| tiny     | ~1GB         | ~30s            | Low       |
| base     | ~1.5GB       | ~45s            | Medium    |
| small    | ~2.5GB       | ~60s            | Good      |
| medium   | ~4GB         | ~90s            | Very Good |
| large-v2 | ~6GB         | ~120s           | Excellent |

### Optimization Tips

1. **For Speed**: Use smaller models (tiny, base)
2. **For Quality**: Use larger models (large-v2, large-v3)
3. **For Memory**: Use float16 compute type
4. **For Concurrency**: Monitor GPU memory usage

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [Faster-Whisper](https://github.com/guillaumekln/faster-whisper) for efficient transcription
- [yt-dlp](https://github.com/yt-dlp/yt-dlp) for video downloading
- [MoviePy](https://github.com/Zulko/moviepy) for video processing
- [Google Colab](https://colab.research.google.com/) for GPU access

## 📞 Support

For support and questions:

- Create an issue on GitHub
- Check the troubleshooting section
- Review the documentation

---

**Note**: This system is designed for educational and research purposes. Please respect YouTube's terms of service and copyright laws when downloading videos.
