import base64
import json
import os
import platform
import queue
import subprocess
import sys
import threading
import time

import requests
from PyQt5.QtCore import pyqtSignal, QObject
from PyQt5.QtWidgets import (QApplication, QMainWindow, QPushButton, QVBoxLayout, QHBoxLayout,
                             QWidget, QFileDialog, QLabel, QProgressBar, QComboBox, QListWidget,
                             QListWidgetItem, QMessageBox,
                             QLineEdit, QTextEdit, QGroupBox)


class WorkerSignals(QObject):
    """Defines the signals available from a running worker thread."""
    progress = pyqtSignal(int)
    finished = pyqtSignal()
    result = pyqtSignal(object)
    error = pyqtSignal(str)
    log = pyqtSignal(str)

class ImageProcessor(threading.Thread):
    """Thread for processing images with Ollama."""
    def __init__(self, job_queue, signals, ollama_url, model):
        super().__init__()
        self.job_queue = job_queue
        self.signals = signals
        self.ollama_url = ollama_url
        self.model = model
        self.running = True
        self.exiftool_path = self.find_exiftool()

    def find_exiftool(self):
        """Find exiftool based on the operating system."""
        if platform.system() == "Darwin":  # macOS
            default_path = "/usr/local/bin/exiftool"
            if os.path.exists(default_path):
                return default_path
            # Try homebrew location
            homebrew_path = "/opt/homebrew/bin/exiftool"
            if os.path.exists(homebrew_path):
                return homebrew_path
        elif platform.system() == "Windows":
            # Check if exiftool is in PATH
            try:
                result = subprocess.run(["where", "exiftool"], 
                                        capture_output=True, 
                                        text=True, 
                                        check=True)
                if result.stdout.strip():
                    return result.stdout.strip().split('\n')[0]
            except subprocess.CalledProcessError:
                pass

        # Try to find in current directory
        current_dir = os.path.dirname(os.path.abspath(__file__))
        exiftool_path = os.path.join(current_dir, "exiftool")
        if os.path.exists(exiftool_path):
            return exiftool_path

        return None

    def process_image(self, image_path):
        """Process a single image with Ollama and write metadata."""
        if not self.exiftool_path:
            self.signals.error.emit("ExifTool not found. Please install ExifTool.")
            self.signals.log.emit("ERROR: ExifTool not found. Please install ExifTool.")
            return False

        try:
            self.signals.log.emit(f"Processing image: {os.path.basename(image_path)}")

            # Read the image and encode it as base64
            self.signals.log.emit("Reading and encoding image...")
            with open(image_path, "rb") as img_file:
                base64_image = base64.b64encode(img_file.read()).decode("utf-8")

            # Prepare the request to Ollama
            prompt = "Generate 10 hashtags and a short paragraph description for this image. Format the response as JSON with 'hashtags' as an array and 'description' as a string."

            payload = {
                "model": self.model,
                "prompt": prompt,
                "images": [base64_image],
                "stream": False
            }

            # Send request to Ollama
            self.signals.log.emit(f"Sending request to Ollama API at {self.ollama_url}...")
            response = requests.post(f"{self.ollama_url}/api/generate", json=payload)
            response.raise_for_status()

            # Parse the response
            self.signals.log.emit("Received response from Ollama, parsing...")
            result = response.json()
            response_text = result.get("response", "")

            # Try to extract JSON from the response
            try:
                # Find JSON in the response (it might be embedded in text)
                json_start = response_text.find("{")
                json_end = response_text.rfind("}") + 1
                if json_start >= 0 and json_end > json_start:
                    json_str = response_text[json_start:json_end]
                    data = json.loads(json_str)
                else:
                    # If no JSON found, try to parse manually
                    hashtags = []
                    description = ""

                    lines = response_text.split('\n')
                    hashtag_mode = False
                    desc_mode = False

                    for line in lines:
                        line = line.strip()
                        if "hashtag" in line.lower() or "#" in line:
                            hashtag_mode = True
                            if "#" in line:
                                tags = [tag.strip() for tag in line.split() if tag.startswith("#")]
                                hashtags.extend(tags)
                        elif "description" in line.lower():
                            hashtag_mode = False
                            desc_mode = True
                        elif hashtag_mode and line:
                            if "#" in line:
                                tags = [tag.strip() for tag in line.split() if tag.startswith("#")]
                                hashtags.extend(tags)
                        elif desc_mode and line:
                            description += line + " "

                    data = {
                        "hashtags": hashtags[:10],  # Limit to 10 hashtags
                        "description": description.strip()
                    }
            except Exception as e:
                # If JSON parsing fails, create a simple structure
                data = {
                    "hashtags": [],
                    "description": response_text
                }

            # Ensure we have hashtags and description
            if "hashtags" not in data:
                data["hashtags"] = []
            if "description" not in data:
                data["description"] = ""

            # Make sure hashtags start with #
            hashtags = data["hashtags"]
            formatted_hashtags = []
            for tag in hashtags:
                if not tag.startswith("#"):
                    tag = "#" + tag
                formatted_hashtags.append(tag)

            # Join hashtags into a string
            hashtags_str = " ".join(formatted_hashtags)
            description = data["description"]

            # Write metadata using exiftool
            self.signals.log.emit("Writing metadata to image using ExifTool...")
            cmd = [
                self.exiftool_path,
                "-overwrite_original",
                f"-IPTC:Keywords={hashtags_str}",
                f"-IPTC:Caption-Abstract={description}",
                image_path
            ]

            cmd_str = " ".join(cmd)
            self.signals.log.emit(f"Running command: {cmd_str}")

            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            self.signals.log.emit(f"ExifTool output: {result.stdout}")
            self.signals.log.emit(f"Successfully processed {os.path.basename(image_path)}")
            return True

        except Exception as e:
            error_msg = f"Error processing {os.path.basename(image_path)}: {str(e)}"
            self.signals.error.emit(error_msg)
            self.signals.log.emit(f"ERROR: {error_msg}")
            return False

    def run(self):
        """Process images from the queue."""
        while self.running:
            try:
                job = self.job_queue.get(timeout=1)
                if job is None:  # Sentinel value to stop the thread
                    self.signals.log.emit("Received stop signal, shutting down worker thread")
                    break

                images, total_count = job
                processed = 0

                self.signals.log.emit(f"Starting to process {total_count} images")

                for img_path in images:
                    if not self.running:
                        self.signals.log.emit("Processing interrupted")
                        break

                    self.signals.log.emit(f"Processing image {processed+1} of {total_count}: {os.path.basename(img_path)}")
                    success = self.process_image(img_path)
                    processed += 1
                    progress = int((processed / total_count) * 100)
                    self.signals.progress.emit(progress)
                    self.signals.log.emit(f"Progress: {progress}%")

                self.signals.log.emit("All images processed successfully")
                self.signals.finished.emit()
                self.job_queue.task_done()

            except queue.Empty:
                continue
            except Exception as e:
                error_msg = f"Processing error: {str(e)}"
                self.signals.error.emit(error_msg)
                self.signals.log.emit(f"ERROR: {error_msg}")
                self.job_queue.task_done()

    def stop(self):
        """Stop the processing thread."""
        self.running = False

class ImageTaggerApp(QMainWindow):
    """Main application window."""
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Image Tagger with Ollama")
        self.setGeometry(100, 100, 1000, 700)  # Increased size to accommodate new UI elements

        self.image_paths = []
        self.selected_images = []
        self.ollama_url = "http://localhost:11434"  # Default to localhost

        # Create a job queue and worker signals
        self.job_queue = queue.Queue()
        self.worker_signals = WorkerSignals()

        # Set up the UI
        self.init_ui()

        # Start the worker thread
        self.processor = ImageProcessor(
            self.job_queue, 
            self.worker_signals, 
            self.ollama_url_input.text(),  # Use the URL from the input field
            self.model_selector.currentText()
        )
        self.processor.daemon = True
        self.processor.start()

        # Connect signals
        self.worker_signals.progress.connect(self.update_progress)
        self.worker_signals.finished.connect(self.processing_finished)
        self.worker_signals.error.connect(self.show_error)
        self.worker_signals.log.connect(self.add_log)

    def init_ui(self):
        """Initialize the user interface."""
        # Main layout
        main_layout = QVBoxLayout()

        # Ollama URL input and connection test
        url_layout = QHBoxLayout()
        url_layout.addWidget(QLabel("Ollama URL:"))
        self.ollama_url_input = QLineEdit(self.ollama_url)
        url_layout.addWidget(self.ollama_url_input, 1)

        self.test_connection_btn = QPushButton("Test Connection")
        self.test_connection_btn.clicked.connect(self.test_connection)
        url_layout.addWidget(self.test_connection_btn)

        main_layout.addLayout(url_layout)

        # Top controls
        top_controls = QHBoxLayout()

        # Folder selection
        self.folder_btn = QPushButton("Load Folder")
        self.folder_btn.clicked.connect(self.load_folder)
        self.folder_label = QLabel("No folder selected")

        top_controls.addWidget(self.folder_btn)
        top_controls.addWidget(self.folder_label, 1)

        # Model selection
        model_layout = QHBoxLayout()
        model_layout.addWidget(QLabel("Model:"))
        self.model_selector = QComboBox()
        self.model_selector.addItems(["qwen2.5:32b", "llama3.2-vision:11b", "qwen2.5vl:7b", "llama3.2:3b", "gemma3:12b"])
        self.model_selector.currentTextChanged.connect(self.model_changed)
        model_layout.addWidget(self.model_selector)

        top_controls.addLayout(model_layout)

        main_layout.addLayout(top_controls)

        # Split the main area into two parts: image list and log window
        split_layout = QHBoxLayout()

        # Image list area
        image_group = QGroupBox("Images")
        image_layout = QVBoxLayout()
        self.image_list = QListWidget()
        self.image_list.setSelectionMode(QListWidget.MultiSelection)
        image_layout.addWidget(self.image_list)
        image_group.setLayout(image_layout)
        split_layout.addWidget(image_group, 2)  # 2:1 ratio

        # Log window
        log_group = QGroupBox("Log")
        log_layout = QVBoxLayout()
        self.log_window = QTextEdit()
        self.log_window.setReadOnly(True)
        self.log_window.setLineWrapMode(QTextEdit.NoWrap)
        log_layout.addWidget(self.log_window)
        log_group.setLayout(log_layout)
        split_layout.addWidget(log_group, 1)  # 2:1 ratio

        main_layout.addLayout(split_layout)

        # Process button and progress bar
        bottom_controls = QHBoxLayout()

        self.process_btn = QPushButton("Process Selected Images")
        self.process_btn.clicked.connect(self.process_images)
        self.process_btn.setEnabled(False)

        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)

        bottom_controls.addWidget(self.process_btn)
        bottom_controls.addWidget(self.progress_bar)

        main_layout.addLayout(bottom_controls)

        # Set the main layout
        central_widget = QWidget()
        central_widget.setLayout(main_layout)
        self.setCentralWidget(central_widget)

    def load_folder(self):
        """Open a folder dialog and load images."""
        folder_path = QFileDialog.getExistingDirectory(self, "Select Folder")
        if not folder_path:
            return

        self.folder_label.setText(folder_path)
        self.image_paths = []
        self.image_list.clear()

        # Find all image files in the folder
        valid_extensions = ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.webp']
        for file in os.listdir(folder_path):
            file_path = os.path.join(folder_path, file)
            if os.path.isfile(file_path):
                ext = os.path.splitext(file)[1].lower()
                if ext in valid_extensions:
                    self.image_paths.append(file_path)
                    item = QListWidgetItem(file)
                    self.image_list.addItem(item)

        if self.image_paths:
            self.process_btn.setEnabled(True)
        else:
            self.folder_label.setText(f"{folder_path} (No images found)")
            self.process_btn.setEnabled(False)

    def process_images(self):
        """Process the selected images."""
        selected_items = self.image_list.selectedItems()
        if not selected_items:
            QMessageBox.warning(self, "No Selection", "Please select at least one image to process.")
            return

        # Get the URL from the input field
        url = self.ollama_url_input.text().strip()
        if not url:
            QMessageBox.warning(self, "Invalid URL", "Please enter a valid Ollama URL")
            return

        # Update the URL
        self.ollama_url = url

        # Check if we need to update the processor's URL
        if self.processor.ollama_url != url:
            self.add_log(f"Updating Ollama URL to: {url}")
            self.processor.ollama_url = url

        # Get the selected image paths
        self.selected_images = []
        for item in selected_items:
            index = self.image_list.row(item)
            self.selected_images.append(self.image_paths[index])

        # Disable controls during processing
        self.folder_btn.setEnabled(False)
        self.process_btn.setEnabled(False)
        self.image_list.setEnabled(False)
        self.ollama_url_input.setEnabled(False)
        self.test_connection_btn.setEnabled(False)
        self.progress_bar.setValue(0)

        self.add_log(f"Starting to process {len(self.selected_images)} images with model: {self.model_selector.currentText()}")

        # Add job to queue
        self.job_queue.put((self.selected_images, len(self.selected_images)))

    def update_progress(self, value):
        """Update the progress bar."""
        self.progress_bar.setValue(value)

    def processing_finished(self):
        """Handle completion of processing."""
        # Re-enable controls
        self.folder_btn.setEnabled(True)
        self.process_btn.setEnabled(True)
        self.image_list.setEnabled(True)
        self.ollama_url_input.setEnabled(True)
        self.test_connection_btn.setEnabled(True)

        self.add_log("Processing complete!")
        QMessageBox.information(self, "Processing Complete", 
                               f"Successfully processed {len(self.selected_images)} images.")

    def show_error(self, message):
        """Display an error message."""
        QMessageBox.critical(self, "Error", message)

    def add_log(self, message):
        """Add a message to the log window."""
        timestamp = time.strftime("%H:%M:%S")
        self.log_window.append(f"[{timestamp}] {message}")
        # Auto-scroll to the bottom
        scrollbar = self.log_window.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())

    def model_changed(self, model_name):
        """Handle model selection change."""
        if hasattr(self, 'processor'):
            self.processor.model = model_name
            self.add_log(f"Model changed to: {model_name}")

    def test_connection(self):
        """Test the connection to the Ollama API."""
        url = self.ollama_url_input.text().strip()
        if not url:
            QMessageBox.warning(self, "Invalid URL", "Please enter a valid Ollama URL")
            return

        # Update the URL
        self.ollama_url = url

        # Disable the test button during the test
        self.test_connection_btn.setEnabled(False)
        self.test_connection_btn.setText("Testing...")

        try:
            self.add_log(f"Testing connection to Ollama at {url}...")

            # Try to connect to the Ollama API
            response = requests.get(f"{url}/api/tags", timeout=5)

            if response.status_code == 200:
                models = response.json().get("models", [])
                model_names = [m.get("name", "") for m in models]

                if model_names:
                    self.add_log(f"Connection successful! Available models: {', '.join(model_names)}")
                    QMessageBox.information(self, "Connection Successful", 
                                          f"Successfully connected to Ollama at {url}.\nAvailable models: {', '.join(model_names)}")
                else:
                    self.add_log("Connection successful, but no models found.")
                    QMessageBox.information(self, "Connection Successful", 
                                          f"Successfully connected to Ollama at {url}, but no models were found.\nYou may need to pull a model first.")
            else:
                error_msg = f"Connection failed with status code {response.status_code}: {response.text}"
                self.add_log(f"ERROR: {error_msg}")
                QMessageBox.critical(self, "Connection Failed", error_msg)

        except requests.exceptions.ConnectionError:
            error_msg = f"Could not connect to Ollama at {url}. Make sure the server is running and accessible."
            self.add_log(f"ERROR: {error_msg}")
            QMessageBox.critical(self, "Connection Failed", error_msg)

        except Exception as e:
            error_msg = f"Error testing connection: {str(e)}"
            self.add_log(f"ERROR: {error_msg}")
            QMessageBox.critical(self, "Connection Failed", error_msg)

        finally:
            # Re-enable the test button
            self.test_connection_btn.setEnabled(True)
            self.test_connection_btn.setText("Test Connection")

    def closeEvent(self, event):
        """Handle window close event."""
        # Stop the processor thread
        if hasattr(self, 'processor'):
            self.processor.stop()
            self.job_queue.put(None)  # Add sentinel to ensure thread exits
            self.processor.join(timeout=1)
        event.accept()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = ImageTaggerApp()
    window.show()
    sys.exit(app.exec_())
