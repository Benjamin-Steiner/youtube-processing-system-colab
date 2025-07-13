"""
Google Colab Interface for YouTube Video Processing System
Provides interactive UI for Google Colab environment
"""

import time
import logging
from typing import Dict, List, Any
from IPython.display import display, HTML, clear_output
import ipywidgets as widgets
from ipywidgets import Layout, VBox, HBox, Label, Button, Text, Dropdown, IntSlider, Checkbox, Output, Progress

from ..models.video_processor import VideoProcessor, ProcessingConfig


class ColabInterface:
    """
    Interactive interface for Google Colab
    Provides widgets and real-time monitoring
    """
    
    def __init__(self, processor: VideoProcessor):
        self.processor = processor
        self.logger = logging.getLogger(__name__)
        
        # UI components
        self.video_input = None
        self.model_dropdown = None
        self.concurrent_slider = None
        self.start_button = None
        self.stop_button = None
        self.status_output = None
        self.progress_output = None
        self.results_output = None
        
        # Monitoring
        self.monitoring_active = False
        self.monitoring_thread = None
        
        self._create_ui()
    
    def _create_ui(self):
        """Create the user interface"""
        # Video input
        self.video_input = Text(
            value='',
            placeholder='Enter YouTube video URL',
            description='Video URL:',
            layout=Layout(width='600px')
        )
        
        # Model selection
        self.model_dropdown = Dropdown(
            options=['tiny', 'base', 'small', 'medium', 'large', 'large-v2', 'large-v3'],
            value='large-v2',
            description='Model:',
            layout=Layout(width='300px')
        )
        
        # Concurrency slider
        self.concurrent_slider = IntSlider(
            value=3,
            min=1,
            max=5,
            step=1,
            description='Concurrent:',
            layout=Layout(width='400px')
        )
        
        # Buttons
        self.start_button = Button(
            description='Start Processing',
            button_style='success',
            layout=Layout(width='150px')
        )
        self.start_button.on_click(self._on_start_click)
        
        self.stop_button = Button(
            description='Stop Processing',
            button_style='danger',
            layout=Layout(width='150px'),
            disabled=True
        )
        self.stop_button.on_click(self._on_stop_click)
        
        # Output areas
        self.status_output = Output(layout=Layout(height='200px'))
        self.progress_output = Output(layout=Layout(height='300px'))
        self.results_output = Output(layout=Layout(height='400px'))
        
        # Create layout
        input_section = VBox([
            Label(value='YouTube Video Processing System', style={'font_weight': 'bold', 'font_size': '18px'}),
            HBox([self.video_input]),
            HBox([self.model_dropdown, self.concurrent_slider]),
            HBox([self.start_button, self.stop_button])
        ])
        
        self.ui = VBox([
            input_section,
            Label(value='Status', style={'font_weight': 'bold'}),
            self.status_output,
            Label(value='Progress', style={'font_weight': 'bold'}),
            self.progress_output,
            Label(value='Results', style={'font_weight': 'bold'}),
            self.results_output
        ])
    
    def _on_start_click(self, button):
        """Handle start button click"""
        video_url = self.video_input.value.strip()
        
        if not video_url:
            with self.status_output:
                clear_output()
                display(HTML('<span style="color: red;">Please enter a video URL</span>'))
            return
        
        # Update configuration
        self.processor.config.whisper_model = self.model_dropdown.value
        self.processor.config.max_concurrent = self.concurrent_slider.value
        
        # Add video to queue
        self.processor.add_video(video_url)
        
        # Start processing
        self.processor.start_processing()
        
        # Update UI
        self.start_button.disabled = True
        self.stop_button.disabled = False
        self.video_input.disabled = True
        
        # Start monitoring
        self._start_monitoring()
        
        with self.status_output:
            clear_output()
            display(HTML(f'<span style="color: green;">Started processing: {video_url}</span>'))
    
    def _on_stop_click(self, button):
        """Handle stop button click"""
        self.processor.stop_processing()
        self._stop_monitoring()
        
        # Update UI
        self.start_button.disabled = False
        self.stop_button.disabled = True
        self.video_input.disabled = False
        
        with self.status_output:
            clear_output()
            display(HTML('<span style="color: orange;">Processing stopped</span>'))
    
    def _start_monitoring(self):
        """Start monitoring thread"""
        self.monitoring_active = True
        import threading
        self.monitoring_thread = threading.Thread(target=self._monitor_progress)
        self.monitoring_thread.daemon = True
        self.monitoring_thread.start()
    
    def _stop_monitoring(self):
        """Stop monitoring thread"""
        self.monitoring_active = False
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=5.0)
    
    def _monitor_progress(self):
        """Monitor processing progress"""
        while self.monitoring_active:
            try:
                status = self.processor.get_status()
                self._update_progress_display(status)
                time.sleep(2)
            except Exception as e:
                self.logger.error(f"Error in monitoring: {e}")
                break
    
    def _update_progress_display(self, status: Dict[str, Any]):
        """Update progress display"""
        with self.progress_output:
            clear_output()
            
            # Create status HTML
            html = f"""
            <div style="font-family: monospace;">
                <h4>Processing Status</h4>
                <table style="width: 100%; border-collapse: collapse;">
                    <tr>
                        <td style="padding: 5px; border: 1px solid #ddd;"><strong>Queue Length:</strong></td>
                        <td style="padding: 5px; border: 1px solid #ddd;">{status['queue_length']}</td>
                    </tr>
                    <tr>
                        <td style="padding: 5px; border: 1px solid #ddd;"><strong>Active Processes:</strong></td>
                        <td style="padding: 5px; border: 1px solid #ddd;">{status['active_processes']}</td>
                    </tr>
                    <tr>
                        <td style="padding: 5px; border: 1px solid #ddd;"><strong>Completed:</strong></td>
                        <td style="padding: 5px; border: 1px solid #ddd; color: green;">{status['completed_count']}</td>
                    </tr>
                    <tr>
                        <td style="padding: 5px; border: 1px solid #ddd;"><strong>Failed:</strong></td>
                        <td style="padding: 5px; border: 1px solid #ddd; color: red;">{status['failed_count']}</td>
                    </tr>
                </table>
                
                <h4>Resource Status</h4>
                <table style="width: 100%; border-collapse: collapse;">
                    <tr>
                        <td style="padding: 5px; border: 1px solid #ddd;"><strong>GPU Memory:</strong></td>
                        <td style="padding: 5px; border: 1px solid #ddd;">{status['resource_status']['gpu']['memory_usage_percent']:.1f}%</td>
                    </tr>
                    <tr>
                        <td style="padding: 5px; border: 1px solid #ddd;"><strong>CPU Usage:</strong></td>
                        <td style="padding: 5px; border: 1px solid #ddd;">{status['resource_status']['cpu']['usage_percent']:.1f}%</td>
                    </tr>
                    <tr>
                        <td style="padding: 5px; border: 1px solid #ddd;"><strong>System Memory:</strong></td>
                        <td style="padding: 5px; border: 1px solid #ddd;">{status['resource_status']['system_memory']['usage_percent']:.1f}%</td>
                    </tr>
                </table>
                
                <h4>Concurrency Status</h4>
                <table style="width: 100%; border-collapse: collapse;">
                    <tr>
                        <td style="padding: 5px; border: 1px solid #ddd;"><strong>Mode:</strong></td>
                        <td style="padding: 5px; border: 1px solid #ddd;">{status['concurrency_status']['mode']}</td>
                    </tr>
                    <tr>
                        <td style="padding: 5px; border: 1px solid #ddd;"><strong>Current Concurrent:</strong></td>
                        <td style="padding: 5px; border: 1px solid #ddd;">{status['concurrency_status']['current_concurrent']}</td>
                    </tr>
                    <tr>
                        <td style="padding: 5px; border: 1px solid #ddd;"><strong>Active Processes:</strong></td>
                        <td style="padding: 5px; border: 1px solid #ddd;">{status['concurrency_status']['active_processes']}</td>
                    </tr>
                </table>
            </div>
            """
            
            display(HTML(html))
    
    def _update_results_display(self):
        """Update results display"""
        results = self.processor.get_results()
        
        with self.results_output:
            clear_output()
            
            html = "<h4>Processing Results</h4>"
            
            if results['completed']:
                html += "<h5>Successfully Processed:</h5>"
                html += "<table style='width: 100%; border-collapse: collapse;'>"
                html += "<tr><th style='padding: 5px; border: 1px solid #ddd;'>Title</th><th style='padding: 5px; border: 1px solid #ddd;'>Time</th><th style='padding: 5px; border: 1px solid #ddd;'>Files</th></tr>"
                
                for result in results['completed']:
                    html += f"""
                    <tr>
                        <td style='padding: 5px; border: 1px solid #ddd;'>{result.video_title}</td>
                        <td style='padding: 5px; border: 1px solid #ddd;'>{result.processing_time:.1f}s</td>
                        <td style='padding: 5px; border: 1px solid #ddd;'>
                            Video: ✓<br>
                            Audio: ✓<br>
                            Transcription: ✓
                        </td>
                    </tr>
                    """
                html += "</table>"
            
            if results['failed']:
                html += "<h5>Failed:</h5>"
                html += "<table style='width: 100%; border-collapse: collapse;'>"
                html += "<tr><th style='padding: 5px; border: 1px solid #ddd;'>URL</th><th style='padding: 5px; border: 1px solid #ddd;'>Error</th></tr>"
                
                for result in results['failed']:
                    html += f"""
                    <tr>
                        <td style='padding: 5px; border: 1px solid #ddd;'>{result.video_url}</td>
                        <td style='padding: 5px; border: 1px solid #ddd; color: red;'>{result.error_message}</td>
                    </tr>
                    """
                html += "</table>"
            
            if not results['completed'] and not results['failed']:
                html += "<p>No results yet...</p>"
            
            display(HTML(html))
    
    def run(self):
        """Run the interface"""
        display(self.ui)
        
        # Start periodic results update
        import threading
        def update_results():
            while self.monitoring_active:
                try:
                    self._update_results_display()
                    time.sleep(5)
                except Exception as e:
                    self.logger.error(f"Error updating results: {e}")
                    break
        
        results_thread = threading.Thread(target=update_results)
        results_thread.daemon = True
        results_thread.start()
    
    def get_recommendations(self) -> List[str]:
        """Get system recommendations"""
        return self.processor.get_recommendations()
    
    def show_recommendations(self):
        """Display system recommendations"""
        recommendations = self.get_recommendations()
        
        if recommendations:
            html = "<h4>System Recommendations</h4><ul>"
            for rec in recommendations:
                html += f"<li>{rec}</li>"
            html += "</ul>"
            
            display(HTML(html))
        else:
            display(HTML("<h4>System Recommendations</h4><p>No recommendations at this time.</p>"))


class ColabSetup:
    """
    Setup utilities for Google Colab
    """
    
    @staticmethod
    def install_dependencies():
        """Install required dependencies"""
        import subprocess
        import sys
        
        packages = [
            'yt-dlp',
            'faster-whisper',
            'moviepy',
            'librosa',
            'soundfile',
            'ipywidgets',
            'GPUtil',
            'psutil'
        ]
        
        for package in packages:
            try:
                subprocess.check_call([sys.executable, '-m', 'pip', 'install', package])
                print(f"✓ Installed {package}")
            except subprocess.CalledProcessError:
                print(f"✗ Failed to install {package}")
    
    @staticmethod
    def setup_gpu():
        """Setup GPU for processing"""
        try:
            import torch
            if torch.cuda.is_available():
                print(f"✓ GPU available: {torch.cuda.get_device_name(0)}")
                print(f"✓ GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f}GB")
            else:
                print("⚠ GPU not available - processing will be slower")
        except ImportError:
            print("⚠ PyTorch not available")
    
    @staticmethod
    def setup_drive():
        """Setup Google Drive integration"""
        try:
            from google.colab import drive
            drive.mount('/content/drive')
            print("✓ Google Drive mounted")
        except ImportError:
            print("⚠ Google Drive not available")
    
    @staticmethod
    def run_setup():
        """Run complete setup"""
        print("Setting up YouTube Video Processing System...")
        
        ColabSetup.install_dependencies()
        ColabSetup.setup_gpu()
        ColabSetup.setup_drive()
        
        print("Setup complete!") 