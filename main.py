import base64
import os
import platform
import queue
import subprocess
import sys
import threading

import requests
from PIL import Image
from PyQt5.QtCore import Qt, pyqtSignal, QObject
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtWidgets import (QApplication, QMainWindow, QPushButton, QFileDialog,
                             QVBoxLayout, QHBoxLayout, QWidget, QLabel, QProgressBar,
                             QListWidget, QListWidgetItem, QComboBox, QMessageBox)


class WorkerSignals(QObject):
    """
    Defines the signals available from a running worker thread.
    """
    progress = pyqtSignal(int)
    finished = pyqtSignal()
    result = pyqtSignal(object)
    error = pyqtSignal(str)

class ImageProcessor(threading.Thread):
    """
    Worker thread for processing images through ollama.
    """
    def __init__(self, image_queue, signals, ollama_model, exiftool_path):
        super().__init__()
        self.image_queue = image_queue
        self.signals = signals
        self.ollama_model = ollama_model
        self.exiftool_path = exiftool_path
        self.running = True
        
    def run(self):
        total_images = self.image_queue.qsize()
        processed = 0
        
        while self.running:
            try:
                if self.image_queue.empty():
                    break
                
                image_path = self.image_queue.get(timeout=1)
                
                # Process the image with ollama
                hashtags, description = self.process_image_with_ollama(image_path)
                
                # Write metadata to image using exiftool
                self.write_metadata_to_image(image_path, hashtags, description)
                
                processed += 1
                progress = int((processed / total_images) * 100)
                self.signals.progress.emit(progress)
                self.image_queue.task_done()
                
            except queue.Empty:
                break
            except Exception as e:
                self.signals.error.emit(str(e))
                break
        
        self.signals.finished.emit()
    
    def process_image_with_ollama(self, image_path):
        """
        Process an image through ollama API to get hashtags and description.
        """
        try:
            with open(image_path, "rb") as image_file:
                image_data = base64.b64encode(image_file.read()).decode('utf-8')
            
            prompt = "Generate 10 hashtags and a short paragraph description for this image."
            
            response = requests.post(
                "http://localhost:11434/api/generate",
                json={
                    "model": self.ollama_model,
                    "prompt": prompt,
                    "images": [image_data],
                    "stream": False
                }
            )
            
            if response.status_code == 200:
                result = response.json()
                response_text = result.get('response', '')
                
                # Parse the response to extract hashtags and description
                lines = response_text.strip().split('\n')
                hashtags = []
                description = ""
                
                for line in lines:
                    if line.startswith('#'):
                        hashtags.append(line.strip())
                    elif line and not line.startswith('#'):
                        if description:
                            description += " " + line.strip()
                        else:
                            description = line.strip()
                
                return hashtags, description
            else:
                raise Exception(f"Error from ollama API: {response.text}")
        
        except Exception as e:
            raise Exception(f"Failed to process image with ollama: {str(e)}")
    
    def write_metadata_to_image(self, image_path, hashtags, description):
        """
        Write hashtags and description to image metadata using exiftool.
        """
        try:
            # Combine hashtags into a single string
            hashtags_str = " ".join(hashtags)
            
            # Use exiftool to write metadata
            cmd = [
                self.exiftool_path,
                "-overwrite_original",
                f"-IPTC:Keywords={hashtags_str}",
                f"-IPTC:Caption-Abstract={description}",
                image_path
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode != 0:
                raise Exception(f"Exiftool error: {result.stderr}")
                
        except Exception as e:
            raise Exception(f"Failed to write metadata: {str(e)}")
    
    def stop(self):
        self.running = False

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        
        self.setWindowTitle("Image Hashtag Generator")
        self.setGeometry(100, 100, 800, 600)
        
        self.image_queue = queue.Queue()
        self.image_paths = []
        self.selected_images = []
        self.exiftool_path = self.find_exiftool()
        
        if not self.exiftool_path:
            QMessageBox.critical(self, "Error", "ExifTool not found. Please install ExifTool and try again.")
            sys.exit(1)
        
        self.init_ui()
    
    def init_ui(self):
        # Main layout
        main_layout = QVBoxLayout()
        
        # Model selection
        model_layout = QHBoxLayout()
        model_label = QLabel("Select Ollama Model:")
        self.model_combo = QComboBox()
        self.model_combo.addItems(["llava", "bakllava", "llava-13b", "llava-34b"])
        model_layout.addWidget(model_label)
        model_layout.addWidget(self.model_combo)
        model_layout.addStretch()
        main_layout.addLayout(model_layout)
        
        # Folder selection
        folder_layout = QHBoxLayout()
        self.folder_label = QLabel("No folder selected")
        folder_button = QPushButton("Load Folder")
        folder_button.clicked.connect(self.select_folder)
        folder_layout.addWidget(self.folder_label)
        folder_layout.addWidget(folder_button)
        main_layout.addLayout(folder_layout)
        
        # Image list
        self.image_list = QListWidget()
        self.image_list.setSelectionMode(QListWidget.MultiSelection)
        main_layout.addWidget(self.image_list)
        
        # Process button
        process_layout = QHBoxLayout()
        process_button = QPushButton("Process Selected Images")
        process_button.clicked.connect(self.process_images)
        process_layout.addStretch()
        process_layout.addWidget(process_button)
        main_layout.addLayout(process_layout)
        
        # Progress bar
        progress_layout = QHBoxLayout()
        progress_label = QLabel("Progress:")
        self.progress_bar = QProgressBar()
        self.progress_bar.setValue(0)
        progress_layout.addWidget(progress_label)
        progress_layout.addWidget(self.progress_bar)
        main_layout.addLayout(progress_layout)
        
        # Status label
        self.status_label = QLabel("Ready")
        main_layout.addWidget(self.status_label)
        
        # Set the main layout
        central_widget = QWidget()
        central_widget.setLayout(main_layout)
        self.setCentralWidget(central_widget)
    
    def find_exiftool(self):
        """
        Find exiftool in the system.
        """
        system = platform.system()
        
        if system == "Darwin":  # macOS
            default_path = "/usr/local/bin/exiftool"
            if os.path.exists(default_path):
                return default_path
            
            # Try to find in other common locations
            for path in ["/opt/homebrew/bin/exiftool", "/usr/bin/exiftool"]:
                if os.path.exists(path):
                    return path
        
        elif system == "Windows":
            # Check if exiftool is in PATH
            try:
                result = subprocess.run(["where", "exiftool"], capture_output=True, text=True)
                if result.returncode == 0:
                    return result.stdout.strip().split("\n")[0]
            except:
                pass
            
            # Check common installation locations
            for path in [
                r"C:\Program Files\ExifTool\exiftool.exe",
                r"C:\Program Files (x86)\ExifTool\exiftool.exe"
            ]:
                if os.path.exists(path):
                    return path
        
        else:  # Linux and others
            try:
                result = subprocess.run(["which", "exiftool"], capture_output=True, text=True)
                if result.returncode == 0:
                    return result.stdout.strip()
            except:
                pass
        
        return None
    
    def select_folder(self):
        """
        Open a dialog to select a folder and scan for images.
        """
        folder = QFileDialog.getExistingDirectory(self, "Select Folder")
        
        if folder:
            self.folder_label.setText(folder)
            self.scan_folder_for_images(folder)
    
    def scan_folder_for_images(self, folder_path):
        """
        Scan the selected folder for image files.
        """
        self.image_list.clear()
        self.image_paths = []
        
        # Common image extensions
        image_extensions = ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.webp']
        
        try:
            for file in os.listdir(folder_path):
                file_path = os.path.join(folder_path, file)
                
                if os.path.isfile(file_path):
                    ext = os.path.splitext(file)[1].lower()
                    
                    if ext in image_extensions:
                        self.image_paths.append(file_path)
                        item = QListWidgetItem(file)
                        self.image_list.addItem(item)
            
            self.status_label.setText(f"Found {len(self.image_paths)} images")
        
        except Exception as e:
            QMessageBox.warning(self, "Error", f"Error scanning folder: {str(e)}")
    
    def process_images(self):
        """
        Process the selected images.
        """
        selected_items = self.image_list.selectedItems()
        
        if not selected_items:
            QMessageBox.warning(self, "Warning", "No images selected")
            return
        
        # Get selected model
        ollama_model = self.model_combo.currentText()
        
        # Clear the queue
        while not self.image_queue.empty():
            try:
                self.image_queue.get_nowait()
            except queue.Empty:
                break
        
        # Add selected images to the queue
        for item in selected_items:
            index = self.image_list.row(item)
            self.image_queue.put(self.image_paths[index])
        
        # Create worker signals
        self.worker_signals = WorkerSignals()
        self.worker_signals.progress.connect(self.update_progress)
        self.worker_signals.finished.connect(self.processing_finished)
        self.worker_signals.error.connect(self.processing_error)
        
        # Start the worker thread
        self.processor = ImageProcessor(
            self.image_queue, 
            self.worker_signals, 
            ollama_model,
            self.exiftool_path
        )
        self.processor.daemon = True
        self.processor.start()
        
        self.status_label.setText("Processing images...")
    
    def update_progress(self, value):
        """
        Update the progress bar.
        """
        self.progress_bar.setValue(value)
    
    def processing_finished(self):
        """
        Called when image processing is complete.
        """
        self.status_label.setText("Processing complete")
        QMessageBox.information(self, "Success", "All images have been processed successfully")
    
    def processing_error(self, error_msg):
        """
        Called when an error occurs during processing.
        """
        self.status_label.setText(f"Error: {error_msg}")
        QMessageBox.critical(self, "Error", f"An error occurred: {error_msg}")
    
    def closeEvent(self, event):
        """
        Clean up when the application is closed.
        """
        if hasattr(self, 'processor') and self.processor.is_alive():
            self.processor.stop()
            self.processor.join(timeout=1)
        event.accept()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())