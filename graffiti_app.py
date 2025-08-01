import sys
import os
import json
import subprocess
import requests
import re
from pathlib import Path
from typing import List, Dict, Optional
import base64
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from enum import Enum

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QGridLayout, QPushButton, QLabel, QFileDialog, QScrollArea,
    QCheckBox, QProgressBar, QStatusBar, QMenuBar, QDialog,
    QFormLayout, QLineEdit, QComboBox, QTextEdit, QSplitter,
    QFrame, QMessageBox, QGroupBox, QSpacerItem, QSizePolicy
)
from PyQt6.QtCore import (
    Qt, QObject, pyqtSignal, QTimer, QRunnable, QThreadPool, pyqtSlot
)
from PyQt6.QtGui import (
    QPixmap, QIcon, QFont, QPalette, QColor, QAction, QCursor
)


class TaskStatus(Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class ProcessingResult:
    image_path: str
    filename: str
    status: TaskStatus
    description: str = ""
    hashtags: str = ""
    error_message: str = ""


class Settings:
    def __init__(self):
        self.settings_file = Path.home() / '.graffiti_settings.json'
        self.load_settings()

    def load_settings(self):
        if self.settings_file.exists():
            with open(self.settings_file, 'r') as f:
                data = json.load(f)
        else:
            data = {}

        self.llm_provider = data.get('llm_provider', 'ollama')
        self.ollama_url = data.get('ollama_url', 'http://localhost:11434')
        self.ollama_model = data.get('ollama_model', 'llava')
        self.lmstudio_url = data.get('lmstudio_url', 'http://localhost:1234')
        self.lmstudio_model = data.get('lmstudio_model', 'llava')
        self.openai_api_key = data.get('openai_api_key', '')
        self.anthropic_api_key = data.get('anthropic_api_key', '')

        # Single comprehensive prompt that gets both description and hashtags
        self.comprehensive_prompt = data.get('comprehensive_prompt',
                                             '''Analyze this image and provide EXACTLY what I request:
                                 
                                 1. DESCRIPTION: Write 2-3 sentences describing the image including the vibe, people, surroundings, and atmosphere.
                                 
                                 2. HASHTAGS: Generate EXACTLY 12-15 relevant hashtags. Each hashtag should be a single word or compound word (no spaces). Focus on: location types, objects, people, mood, colors, time of day, weather, activities, and style.
                                 
                                 Format your response EXACTLY like this:
                                 DESCRIPTION: [your 2-3 sentence description here]
                                 HASHTAGS: [hashtag1] [hashtag2] [hashtag3] [hashtag4] [hashtag5] [hashtag6] [hashtag7] [hashtag8] [hashtag9] [hashtag10] [hashtag11] [hashtag12] [hashtag13] [hashtag14] [hashtag15]
                                 
                                 Important: Do not include # symbols. Do not number the hashtags. Do not use phrases with spaces. Each hashtag should be one word or compound word only.''')

    def save_settings(self):
        data = {
            'llm_provider': self.llm_provider,
            'ollama_url': self.ollama_url,
            'ollama_model': self.ollama_model,
            'lmstudio_url': self.lmstudio_url,
            'lmstudio_model': self.lmstudio_model,
            'openai_api_key': self.openai_api_key,
            'anthropic_api_key': self.anthropic_api_key,
            'comprehensive_prompt': self.comprehensive_prompt
        }
        with open(self.settings_file, 'w') as f:
            json.dump(data, f, indent=2)


class SettingsDialog(QDialog):
    def __init__(self, settings: Settings, parent=None):
        super().__init__(parent)
        self.settings = settings
        self.setWindowTitle("Graffiti Settings")
        self.setModal(True)
        self.resize(700, 600)
        self.setup_ui()

    def setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(20)
        layout.setContentsMargins(25, 25, 25, 25)

        # LLM Provider Selection
        provider_group = QGroupBox("LLM Provider")
        provider_layout = QFormLayout(provider_group)
        provider_layout.setSpacing(15)

        self.provider_combo = QComboBox()
        self.provider_combo.addItems(['ollama', 'lmstudio', 'openai'])
        self.provider_combo.setCurrentText(self.settings.llm_provider)
        provider_layout.addRow("Provider:", self.provider_combo)

        # Ollama Settings
        ollama_group = QGroupBox("Ollama Settings")
        ollama_layout = QFormLayout(ollama_group)
        ollama_layout.setSpacing(15)

        self.ollama_url_edit = QLineEdit(self.settings.ollama_url)
        self.ollama_model_edit = QLineEdit(self.settings.ollama_model)
        ollama_layout.addRow("URL:", self.ollama_url_edit)
        ollama_layout.addRow("Model:", self.ollama_model_edit)

        # LMStudio Settings
        lmstudio_group = QGroupBox("LMStudio Settings")
        lmstudio_layout = QFormLayout(lmstudio_group)
        lmstudio_layout.setSpacing(15)

        self.lmstudio_url_edit = QLineEdit(self.settings.lmstudio_url)
        self.lmstudio_model_edit = QLineEdit(self.settings.lmstudio_model)
        lmstudio_layout.addRow("URL:", self.lmstudio_url_edit)
        lmstudio_layout.addRow("Model:", self.lmstudio_model_edit)

        # API Keys
        api_group = QGroupBox("API Keys")
        api_layout = QFormLayout(api_group)
        api_layout.setSpacing(15)

        self.openai_key_edit = QLineEdit(self.settings.openai_api_key)
        self.openai_key_edit.setEchoMode(QLineEdit.EchoMode.Password)

        api_layout.addRow("OpenAI API Key:", self.openai_key_edit)

        # Comprehensive Prompt
        prompt_group = QGroupBox("Comprehensive Prompt")
        prompt_layout = QFormLayout(prompt_group)
        prompt_layout.setSpacing(15)

        self.prompt_edit = QTextEdit()
        self.prompt_edit.setPlainText(self.settings.comprehensive_prompt)
        self.prompt_edit.setMinimumHeight(200)

        prompt_layout.addRow("Single Prompt (gets both description and hashtags):", self.prompt_edit)

        # Buttons
        button_layout = QHBoxLayout()
        save_button = QPushButton("Save Settings")
        cancel_button = QPushButton("Cancel")

        save_button.clicked.connect(self.save_settings)
        cancel_button.clicked.connect(self.reject)

        button_layout.addStretch()
        button_layout.addWidget(save_button)
        button_layout.addWidget(cancel_button)

        # Add all groups to main layout
        layout.addWidget(provider_group)
        layout.addWidget(ollama_group)
        layout.addWidget(lmstudio_group)
        layout.addWidget(api_group)
        layout.addWidget(prompt_group)
        layout.addLayout(button_layout)

    def save_settings(self):
        self.settings.llm_provider = self.provider_combo.currentText()
        self.settings.ollama_url = self.ollama_url_edit.text()
        self.settings.ollama_model = self.ollama_model_edit.text()
        self.settings.lmstudio_url = self.lmstudio_url_edit.text()
        self.settings.lmstudio_model = self.lmstudio_model_edit.text()
        self.settings.openai_api_key = self.openai_key_edit.text()
        self.settings.comprehensive_prompt = self.prompt_edit.toPlainText()

        self.settings.save_settings()
        self.accept()


class ImageZoomDialog(QDialog):
    """Dialog for viewing full-size images"""

    def __init__(self, image_path: str, parent=None):
        super().__init__(parent)
        self.image_path = image_path
        self.setWindowTitle(f"Image Viewer - {os.path.basename(image_path)}")
        self.setModal(True)
        self.setup_ui()
        self.load_full_image()

    def setup_ui(self):
        self.setMinimumSize(800, 600)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)

        # Scroll area for large images
        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setAlignment(Qt.AlignmentFlag.AlignCenter)

        # Image label
        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.image_label.setStyleSheet("background-color: #2b2b2b; border: 1px solid #404040;")

        self.scroll_area.setWidget(self.image_label)

        # Info label
        self.info_label = QLabel()
        self.info_label.setStyleSheet("color: #bbbbbb; font-size: 12px; padding: 8px;")
        self.info_label.setAlignment(Qt.AlignmentFlag.AlignCenter)

        # Close button
        close_button = QPushButton("Close")
        close_button.clicked.connect(self.close)
        close_button.setMinimumHeight(35)

        layout.addWidget(self.scroll_area)
        layout.addWidget(self.info_label)
        layout.addWidget(close_button)

    def load_full_image(self):
        try:
            pixmap = QPixmap(self.image_path)
            if not pixmap.isNull():
                # Get image info
                file_size = os.path.getsize(self.image_path) / (1024 * 1024)  # MB
                self.info_label.setText(
                    f"Resolution: {pixmap.width()}×{pixmap.height()} | "
                    f"Size: {file_size:.1f} MB | "
                    f"Format: {os.path.splitext(self.image_path)[1].upper()}"
                )

                # Scale image to fit screen if too large
                screen_size = QApplication.primaryScreen().availableGeometry().size()
                max_width = int(screen_size.width() * 0.8)
                max_height = int(screen_size.height() * 0.8)

                if pixmap.width() > max_width or pixmap.height() > max_height:
                    scaled_pixmap = pixmap.scaled(
                        max_width, max_height,
                        Qt.AspectRatioMode.KeepAspectRatio,
                        Qt.TransformationMode.SmoothTransformation
                    )
                    self.image_label.setPixmap(scaled_pixmap)
                else:
                    self.image_label.setPixmap(pixmap)

                # Resize dialog to fit image
                self.resize(min(pixmap.width() + 50, max_width),
                            min(pixmap.height() + 150, max_height))
            else:
                self.image_label.setText("Could not load image")
                self.info_label.setText("Error loading image file")

        except Exception as e:
            self.image_label.setText("Error loading image")
            self.info_label.setText(f"Error: {str(e)}")


class ImageThumbnail(QWidget):
    def __init__(self, image_path: str, parent=None):
        super().__init__(parent)
        self.image_path = image_path
        self.selected = False
        self.setup_ui()

    def setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(8)

        # Checkbox
        self.checkbox = QCheckBox()
        self.checkbox.stateChanged.connect(self.on_selection_changed)

        # Image label - make it clickable
        self.image_label = QLabel()
        self.image_label.setFixedSize(160, 160)
        self.image_label.setStyleSheet("""
            QLabel {
                border: 2px solid #4a4a4a;
                border-radius: 10px;
                background-color: #2b2b2b;
                cursor: pointer;
            }
            QLabel:hover {
                border: 2px solid #00aaff;
            }
        """)
        self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.image_label.setScaledContents(True)
        self.image_label.setCursor(QCursor(Qt.CursorShape.PointingHandCursor))

        # Make image clickable
        self.image_label.mousePressEvent = self.open_zoom_dialog

        # Load thumbnail
        self.load_thumbnail()

        # File name
        filename = os.path.basename(self.image_path)
        self.name_label = QLabel(filename)
        self.name_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.name_label.setWordWrap(True)
        self.name_label.setMaximumWidth(160)
        self.name_label.setStyleSheet("color: #e0e0e0; font-size: 11px;")

        # Add tooltip
        self.image_label.setToolTip("Click to view full-size image")

        layout.addWidget(self.checkbox, alignment=Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.image_label)
        layout.addWidget(self.name_label)

    def open_zoom_dialog(self, event):
        """Open the image in full-size zoom dialog"""
        dialog = ImageZoomDialog(self.image_path, self.parent())
        dialog.exec()

    def load_thumbnail(self):
        try:
            pixmap = QPixmap(self.image_path)
            if not pixmap.isNull():
                scaled_pixmap = pixmap.scaled(
                    150, 150,
                    Qt.AspectRatioMode.KeepAspectRatio,
                    Qt.TransformationMode.SmoothTransformation
                )
                self.image_label.setPixmap(scaled_pixmap)
            else:
                self.image_label.setText("Invalid Image")
                self.image_label.setStyleSheet("""
                    QLabel {
                        border: 2px solid #4a4a4a;
                        border-radius: 10px;
                        background-color: #2b2b2b;
                        color: #888;
                    }
                """)
        except Exception as e:
            self.image_label.setText("Load Error")
            self.image_label.setStyleSheet("""
                QLabel {
                    border: 2px solid #4a4a4a;
                    border-radius: 10px;
                    background-color: #2b2b2b;
                    color: #888;
                }
            """)

    def on_selection_changed(self, state):
        self.selected = state == Qt.CheckState.Checked.value
        if self.selected:
            self.image_label.setStyleSheet("""
                QLabel {
                    border: 3px solid #00aaff;
                    border-radius: 10px;
                    background-color: #2b2b2b;
                }
            """)
        else:
            self.image_label.setStyleSheet("""
                QLabel {
                    border: 2px solid #4a4a4a;
                    border-radius: 10px;
                    background-color: #2b2b2b;
                }
            """)

    def is_selected(self):
        return self.checkbox.isChecked()


class ProcessingSignals(QObject):
    """Signals for communicating between worker threads and main thread"""
    task_started = pyqtSignal(str)  # filename
    task_completed = pyqtSignal(ProcessingResult)
    all_completed = pyqtSignal(list)  # list of ProcessingResult
    error_occurred = pyqtSignal(str)


class ImageProcessor(QRunnable):
    """Worker thread for processing a single image"""

    def __init__(self, image_path: str, settings: Settings, signals: ProcessingSignals):
        super().__init__()
        self.image_path = image_path
        self.settings = settings
        self.signals = signals
        self.filename = os.path.basename(image_path)
        self.is_cancelled = False

    def run(self):
        """Process a single image with comprehensive prompt"""
        result = ProcessingResult(
            image_path=self.image_path,
            filename=self.filename,
            status=TaskStatus.PROCESSING
        )

        if self.is_cancelled:
            result.status = TaskStatus.CANCELLED
            self.signals.task_completed.emit(result)
            return

        try:
            print(f"[Worker] Starting {self.filename}")
            self.signals.task_started.emit(self.filename)

            # Single request for both description and hashtags
            print(f"[Worker] Calling analyze_image_comprehensive for {self.filename}")
            response = self.analyze_image_comprehensive()

            if self.is_cancelled:
                print(f"[Worker] Cancelled during analysis for {self.filename}")
                result.status = TaskStatus.CANCELLED
                self.signals.task_completed.emit(result)
                return

            if response:
                print(f"[Worker] Got response, parsing for {self.filename}")
                # Parse the comprehensive response
                description, hashtags = self.parse_comprehensive_response(response)

                if description and hashtags:
                    print(f"[Worker] Writing metadata for {self.filename}")
                    # Write metadata
                    self.write_metadata(description, hashtags)
                    result.description = description
                    result.hashtags = hashtags
                    result.status = TaskStatus.COMPLETED
                    print(f"[Worker] ✓ Completed {self.filename}")
                else:
                    print(f"[Worker] ✗ Failed to parse response for {self.filename}")
                    result.status = TaskStatus.FAILED
                    result.error_message = "Failed to parse LLM response"
            else:
                print(f"[Worker] ✗ No response from LLM for {self.filename}")
                result.status = TaskStatus.FAILED
                result.error_message = "No response from LLM"

        except Exception as e:
            print(f"[Worker] ✗ Exception in {self.filename}: {e}")
            result.status = TaskStatus.FAILED
            result.error_message = str(e)

        print(f"[Worker] Emitting completion signal for {self.filename}")
        self.signals.task_completed.emit(result)

    def cancel(self):
        """Cancel this processing task"""
        self.is_cancelled = True

    def analyze_image_comprehensive(self) -> Optional[str]:
        """Send single comprehensive request to LLM with retry logic"""
        max_retries = 3
        base_timeout = 120  # Start with 2 minutes

        for attempt in range(max_retries):
            if self.is_cancelled:
                return None

            try:
                print(f"[Analysis] Attempt {attempt + 1}/{max_retries} for {self.filename}")

                # Clear Ollama memory before processing (especially on retries)
                if attempt > 0:
                    self.clear_ollama_memory()

                image_b64 = self.encode_image_base64()

                # Progressive timeout: 2min, 4min, 6min
                timeout = base_timeout * (attempt + 1)
                print(f"[Analysis] Using {timeout}s timeout for attempt {attempt + 1}")

                if self.settings.llm_provider == 'ollama':
                    return self.request_ollama_with_retry(image_b64, timeout)
                elif self.settings.llm_provider == 'lmstudio':
                    return self.request_lmstudio(image_b64)
                elif self.settings.llm_provider == 'openai':
                    return self.request_openai(image_b64)
                else:
                    raise ValueError(f"Unsupported provider: {self.settings.llm_provider}")

            except Exception as e:
                print(f"[Analysis] Attempt {attempt + 1} failed: {e}")
                if attempt == max_retries - 1:
                    print(f"[Analysis] All attempts failed for {self.filename}")
                    return None
                else:
                    print(f"[Analysis] Retrying in 10 seconds...")
                    time.sleep(10)  # Wait before retry

        return None

    def clear_ollama_memory(self):
        """Clear Ollama's GPU memory by unloading the model"""
        try:
            print(f"[Ollama] Clearing GPU memory...")

            # First, try to unload the model
            unload_payload = {
                "model": self.settings.ollama_model,
                "keep_alive": 0  # Immediately unload
            }

            requests.post(
                f"{self.settings.ollama_url}/api/generate",
                json=unload_payload,
                timeout=30
            )

            # Wait a moment for cleanup
            time.sleep(2)
            print(f"[Ollama] Memory cleared")

        except Exception as e:
            print(f"[Ollama] Warning: Could not clear memory: {e}")

    def check_ollama_health(self) -> bool:
        """Check if Ollama is responsive"""
        try:
            response = requests.get(f"{self.settings.ollama_url}/api/tags", timeout=10)
            return response.ok
        except:
            return False

    def request_ollama_with_retry(self, image_b64: str, timeout: int) -> Optional[str]:
        """Ollama request with health check and memory management"""
        try:
            # Health check first
            if not self.check_ollama_health():
                print(f"[Ollama] Health check failed, Ollama may be unresponsive")
                return None

            payload = {
                "model": self.settings.ollama_model,
                "prompt": self.settings.comprehensive_prompt,
                "images": [image_b64],
                "stream": False,
                "options": {
                    "num_ctx": 4096,  # Limit context to prevent memory issues
                    "num_predict": 512,  # Limit response length
                    "temperature": 0.7
                }
            }

            print(f"[Ollama] Sending request for {self.filename} (timeout: {timeout}s)")

            session = requests.Session()
            response = session.post(
                f"{self.settings.ollama_url}/api/generate",
                json=payload,
                timeout=timeout,
                headers={'Connection': 'close'}
            )

            if response.ok:
                result = response.json().get('response', '').strip()
                print(f"[Ollama] ✓ Got response ({len(result)} chars)")

                # Clear memory after successful request to prevent buildup
                self.clear_ollama_memory()
                return result
            else:
                print(f"[Ollama] ✗ Request failed: {response.status_code} - {response.text}")
                return None

        except requests.exceptions.Timeout:
            print(f"[Ollama] ✗ Request timed out after {timeout}s")
            # Try to clear memory after timeout
            self.clear_ollama_memory()
            return None
        except requests.exceptions.ConnectionError as e:
            print(f"[Ollama] ✗ Connection error: {e}")
            return None
        except Exception as e:
            print(f"[Ollama] ✗ Unexpected error: {e}")
            return None

    def request_lmstudio(self, image_b64: str) -> Optional[str]:
        """Single LMStudio request for comprehensive response"""
        try:
            headers = {"Content-Type": "application/json"}
            payload = {
                "model": self.settings.lmstudio_model,
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": self.settings.comprehensive_prompt},
                            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"}}
                        ]
                    }
                ]
            }

            response = requests.post(
                f"{self.settings.lmstudio_url}/v1/chat/completions",
                headers=headers,
                json=payload,
                timeout=180
            )

            if response.ok:
                return response.json()['choices'][0]['message']['content'].strip()
            else:
                return None

        except requests.exceptions.RequestException as e:
            print(f"[LMStudio] Request error: {e}")
            return None

    def request_openai(self, image_b64: str) -> Optional[str]:
        """Single OpenAI request for comprehensive response"""
        try:
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.settings.openai_api_key}"
            }

            payload = {
                "model": "gpt-4-vision-preview",
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": self.settings.comprehensive_prompt},
                            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"}}
                        ]
                    }
                ],
                "max_tokens": 500
            }

            response = requests.post(
                "https://api.openai.com/v1/chat/completions",
                headers=headers,
                json=payload,
                timeout=180
            )

            if response.ok:
                return response.json()['choices'][0]['message']['content'].strip()
            else:
                return None

        except requests.exceptions.RequestException as e:
            print(f"[OpenAI] Request error: {e}")
            return None

    def parse_comprehensive_response(self, response: str) -> tuple[str, str]:
        """Parse the comprehensive response into description and hashtags with improved parsing"""
        try:
            description = ""
            hashtags = ""

            print(f"[Parser] Raw response: {response[:200]}...")

            # Split into lines for processing
            lines = response.split('\n')
            current_section = None
            description_lines = []
            hashtag_lines = []

            for line in lines:
                line = line.strip()
                if not line:
                    continue

                # Check for section headers (case insensitive)
                line_upper = line.upper()
                if 'DESCRIPTION:' in line_upper:
                    current_section = 'description'
                    # Extract description from same line if present
                    desc_start = line_upper.find('DESCRIPTION:') + 12
                    desc_text = line[desc_start:].strip()
                    if desc_text:
                        description_lines.append(desc_text)
                elif 'HASHTAGS:' in line_upper or 'HASHTAG:' in line_upper:
                    current_section = 'hashtags'
                    # Extract hashtags from same line if present
                    hash_start = max(
                        line_upper.find('HASHTAGS:') + 9 if 'HASHTAGS:' in line_upper else -1,
                        line_upper.find('HASHTAG:') + 8 if 'HASHTAG:' in line_upper else -1
                    )
                    if hash_start > 8:  # Valid start position
                        hash_text = line[hash_start:].strip()
                        if hash_text:
                            hashtag_lines.append(hash_text)
                elif current_section == 'description':
                    description_lines.append(line)
                elif current_section == 'hashtags':
                    hashtag_lines.append(line)
                elif not current_section:
                    # If no section header found yet, try to detect content type
                    if len(line.split()) > 5:  # Likely a description (longer text)
                        description_lines.append(line)
                    elif '#' in line or len(line.split()) <= 5:  # Likely hashtags
                        hashtag_lines.append(line)

            # Combine lines
            if description_lines:
                description = ' '.join(description_lines)
            if hashtag_lines:
                hashtags = ' '.join(hashtag_lines)

            # Fallback parsing if sections weren't clearly identified
            if not description and not hashtags:
                print("[Parser] Fallback parsing - no clear sections found")
                # Split response and try to identify description vs hashtags
                words = response.split()
                if len(words) > 10:
                    # Take first part as description, last part as hashtags
                    split_point = len(words) // 2
                    description = ' '.join(words[:split_point])
                    hashtags = ' '.join(words[split_point:])

            # Clean up the results
            if description:
                description = self.sanitize_description(description)
            if hashtags:
                hashtags = self.sanitize_hashtags(hashtags)

            print(f"[Parser] Parsed description: {description[:50]}...")
            print(f"[Parser] Parsed hashtags: {hashtags[:50]}...")

            return description, hashtags

        except Exception as e:
            print(f"Error parsing comprehensive response: {e}")
            return "", ""

    def sanitize_description(self, description: str) -> str:
        """Clean description response, removing obvious conversational prefixes"""
        if not description:
            return ""

        description = description.strip()

        # Remove common conversational starters
        skip_phrases = [
            'this image shows', 'this image depicts', 'this image contains',
            'in this image', 'the image shows', 'the image depicts',
            'i can see', 'this appears to be', 'this seems to be'
        ]

        # Check if description starts with conversational phrases and remove them
        lower_desc = description.lower()
        for phrase in skip_phrases:
            if lower_desc.startswith(phrase):
                # Remove the phrase and clean up
                description = description[len(phrase):].strip()
                if description.startswith(','):
                    description = description[1:].strip()
                break

        # Capitalize first letter
        if description:
            description = description[0].upper() + description[1:]

        return description

    def sanitize_hashtags(self, hashtag_response: str) -> str:
        """Extract clean hashtags from response with improved parsing"""
        if not hashtag_response:
            return ""

        clean_tags = []

        # First try to extract from brackets if present
        bracket_match = re.search(r'\[([^\]]+)\]', hashtag_response)
        if bracket_match:
            hashtag_response = bracket_match.group(1)

        # Split on various separators
        pattern = r'[,\s\n\[\]]+'
        potential_tags = re.split(pattern, hashtag_response)

        for tag in potential_tags:
            # Clean the tag
            tag = tag.strip('#').strip('[]').strip('"').strip("'").strip()

            # Skip empty tags and short tags
            if not tag or len(tag) < 2:
                continue

            # Skip common words
            skip_words = {'and', 'or', 'the', 'of', 'in', 'on', 'at', 'to', 'for', 'with', 'by'}
            if tag.lower() in skip_words:
                continue

            # Only keep valid hashtag characters
            valid_pattern = r'^[a-zA-Z0-9_-]+$'
            if re.match(valid_pattern, tag) and not tag.isdigit():
                clean_tags.append(tag.lower())

        # Remove duplicates while preserving order
        seen = set()
        unique_tags = []
        for tag in clean_tags:
            if tag not in seen:
                seen.add(tag)
                unique_tags.append(tag)

        # Add generic tags if we don't have enough
        if len(unique_tags) < 10:
            print(f"[Sanitization] Only got {len(unique_tags)} hashtags")
            generic_tags = ['photo', 'image', 'photography', 'capture', 'moment']
            for generic in generic_tags:
                if generic not in unique_tags and len(unique_tags) < 12:
                    unique_tags.append(generic)

        # Take up to 15 hashtags
        final_tags = unique_tags[:15]

        print(f"[Sanitization] Final tags ({len(final_tags)}): {' '.join(final_tags)}")

        return ' '.join(final_tags)

    def encode_image_base64(self) -> str:
        with open(self.image_path, 'rb') as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')

    def write_metadata(self, description: str, hashtags: str):
        """Write metadata with proper error handling and validation"""
        try:
            print(f"Writing metadata to {os.path.basename(self.image_path)}")
            print(f"  Description: {description[:50]}...")
            print(f"  Hashtags: {hashtags[:50]}...")

            # Use exiftool to write metadata
            cmd = [
                'exiftool',
                '-overwrite_original',
                f'-ImageDescription={description}',
                f'-Keywords={hashtags}',
                f'-Caption-Abstract={description}',
                self.image_path
            ]

            result = subprocess.run(cmd, check=True, capture_output=True, text=True)
            print(f"[Metadata] ✓ Successfully wrote to {self.filename}")

        except subprocess.CalledProcessError as e:
            raise Exception(f"ExifTool error: {e.stderr}")
        except Exception as e:
            raise Exception(f"Metadata write error: {str(e)}")


class ProcessingManager(QObject):
    """Manages the processing queue and thread pool"""

    progress_updated = pyqtSignal(int, int, str)  # completed, total, current_filename
    processing_complete = pyqtSignal(list)  # list of ProcessingResult
    error_occurred = pyqtSignal(str)

    def __init__(self):
        super().__init__()
        self.thread_pool = QThreadPool()
        self.thread_pool.setMaxThreadCount(1)  # Process one image at a time
        self.signals = ProcessingSignals()
        self.signals.task_started.connect(self.on_task_started)
        self.signals.task_completed.connect(self.on_task_completed)

        self.image_paths = []
        self.results = []
        self.current_processors = []
        self.is_cancelled = False
        self.delay_timer = QTimer()
        self.delay_timer.setSingleShot(True)
        self.delay_timer.timeout.connect(self.process_next_image)
        self.current_index = 0

        # Add a safety timer to detect stuck processing
        self.safety_timer = QTimer()
        self.safety_timer.setSingleShot(True)
        self.safety_timer.timeout.connect(self.handle_stuck_task)
        self.current_task_start_time = None

    def start_processing(self, image_paths: List[str], settings: Settings):
        """Start processing a list of images"""
        self.image_paths = image_paths
        self.results = []
        self.current_processors = []
        self.is_cancelled = False
        self.current_index = 0
        self.settings = settings  # Store settings for memory management

        if not image_paths:
            self.processing_complete.emit([])
            return

        print(f"[Manager] Starting processing of {len(image_paths)} images with {settings.llm_provider}")

        # Clear Ollama memory before starting if using Ollama
        if settings.llm_provider == 'ollama':
            print(f"[Manager] Clearing Ollama memory before starting...")
            self.clear_ollama_memory_manager()

        self.process_next_image()

    def clear_ollama_memory_manager(self):
        """Clear Ollama memory from the manager level"""
        try:
            unload_payload = {
                "model": self.settings.ollama_model,
                "keep_alive": 0
            }

            requests.post(
                f"{self.settings.ollama_url}/api/generate",
                json=unload_payload,
                timeout=10
            )
            print(f"[Manager] Cleared Ollama memory")
        except Exception as e:
            print(f"[Manager] Could not clear Ollama memory: {e}")

    def process_next_image(self):
        """Process the next image in the queue"""
        if self.is_cancelled or self.current_index >= len(self.image_paths):
            self.finish_processing()
            return

        image_path = self.image_paths[self.current_index]
        processor = ImageProcessor(image_path, self.settings, self.signals)
        self.current_processors.append(processor)

        print(
            f"[Manager] Processing image {self.current_index + 1}/{len(self.image_paths)}: {os.path.basename(image_path)}")

        # Start safety timer (15 minutes max per image with retries)
        self.safety_timer.start(900000)  # 15 minute timeout
        self.current_task_start_time = time.time()

        self.thread_pool.start(processor)

    def handle_stuck_task(self):
        """Handle a task that appears to be stuck"""
        print("[Manager] ⚠️ Task appears to be stuck, forcing completion")

        # Cancel current processors
        for processor in self.current_processors:
            processor.cancel()

        # Create a failed result for the stuck task
        if self.current_index < len(self.image_paths):
            stuck_image = self.image_paths[self.current_index]
            stuck_result = ProcessingResult(
                image_path=stuck_image,
                filename=os.path.basename(stuck_image),
                status=TaskStatus.FAILED,
                error_message="Task timed out (stuck for >15 minutes)"
            )
            self.results.append(stuck_result)
            self.current_index += 1

        # Continue with next image
        if not self.is_cancelled and self.current_index < len(self.image_paths):
            self.delay_timer.start(8000)  # 8 second delay before next
        else:
            self.finish_processing()

    def stop_processing(self):
        """Stop all processing"""
        print("[Manager] Stopping processing...")
        self.is_cancelled = True
        self.delay_timer.stop()
        self.safety_timer.stop()  # Stop safety timer too

        # Cancel all current processors
        for processor in self.current_processors:
            processor.cancel()

        # Wait for thread pool to finish
        self.thread_pool.waitForDone(3000)  # 3 second timeout

        print("[Manager] Processing stopped")

    @pyqtSlot(str)
    def on_task_started(self, filename: str):
        """Called when a task starts"""
        self.progress_updated.emit(self.current_index, len(self.image_paths), filename)

    @pyqtSlot(ProcessingResult)
    def on_task_completed(self, result: ProcessingResult):
        """Called when a task completes"""
        # Stop the safety timer since task completed
        self.safety_timer.stop()

        self.results.append(result)
        self.current_index += 1

        elapsed_time = time.time() - self.current_task_start_time if self.current_task_start_time else 0
        print(f"[Manager] Completed {result.filename} with status {result.status} (took {elapsed_time:.1f}s)")

        if result.status == TaskStatus.COMPLETED:
            self.progress_updated.emit(self.current_index, len(self.image_paths), f"✓ {result.filename}")
        elif result.status == TaskStatus.FAILED:
            self.progress_updated.emit(self.current_index, len(self.image_paths),
                                       f"✗ {result.filename} - {result.error_message}")
        else:
            self.progress_updated.emit(self.current_index, len(self.image_paths), f"◉ {result.filename}")

        if not self.is_cancelled and self.current_index < len(self.image_paths):
            # Add longer delay between images and clear Ollama memory
            print(f"[Manager] Waiting 8 seconds and clearing memory before next image...")

            # Clear Ollama memory between images if using Ollama
            if hasattr(self, 'settings') and self.settings.llm_provider == 'ollama':
                self.clear_ollama_memory_manager()

            self.delay_timer.start(8000)  # 8 second delay for more stability
        else:
            self.finish_processing()

    def finish_processing(self):
        """Finish processing and emit results"""
        completed_results = [r for r in self.results if r.status == TaskStatus.COMPLETED]
        print(f"[Manager] Processing finished. Completed {len(completed_results)}/{len(self.results)} images")
        self.processing_complete.emit(completed_results)


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.settings = Settings()
        self.image_thumbnails = []
        self.current_folder = None
        self.processing_manager = ProcessingManager()
        self.setup_signals()
        self.apply_dark_theme()
        self.setup_ui()
        self.setup_menu()

    def setup_signals(self):
        """Connect processing manager signals"""
        self.processing_manager.progress_updated.connect(self.update_progress)
        self.processing_manager.processing_complete.connect(self.processing_finished)
        self.processing_manager.error_occurred.connect(self.processing_error)

    def apply_dark_theme(self):
        # Comprehensive dark theme
        dark_style = """
        QMainWindow {
            background-color: #1e1e1e;
            color: #e0e0e0;
        }

        QWidget {
            background-color: #1e1e1e;
            color: #e0e0e0;
            font-family: 'Segoe UI', Arial, sans-serif;
            font-size: 13px;
        }

        QPushButton {
            background-color: #0078d4;
            color: white;
            border: none;
            padding: 12px 24px;
            border-radius: 8px;
            font-weight: 600;
            font-size: 14px;
            min-height: 20px;
        }

        QPushButton:hover {
            background-color: #106ebe;
        }

        QPushButton:pressed {
            background-color: #005a9e;
        }

        QPushButton:disabled {
            background-color: #3a3a3a;
            color: #888888;
        }

        QGroupBox {
            font-weight: 600;
            font-size: 14px;
            border: 2px solid #404040;
            border-radius: 10px;
            margin-top: 10px;
            padding-top: 15px;
            background-color: #2b2b2b;
        }

        QGroupBox::title {
            subcontrol-origin: margin;
            left: 15px;
            padding: 0 8px 0 8px;
            color: #00aaff;
        }

        QFrame {
            background-color: #2b2b2b;
            border: 1px solid #404040;
            border-radius: 10px;
        }

        QLabel {
            color: #e0e0e0;
            background-color: transparent;
        }

        QLineEdit, QTextEdit, QComboBox {
            background-color: #383838;
            border: 2px solid #505050;
            border-radius: 6px;
            padding: 8px;
            color: #e0e0e0;
            font-size: 13px;
        }

        QLineEdit:focus, QTextEdit:focus, QComboBox:focus {
            border-color: #00aaff;
        }

        QCheckBox {
            color: #e0e0e0;
            spacing: 8px;
        }

        QCheckBox::indicator {
            width: 18px;
            height: 18px;
            border-radius: 3px;
            border: 2px solid #505050;
            background-color: #383838;
        }

        QCheckBox::indicator:checked {
            background-color: #00aaff;
            border-color: #00aaff;
        }

        QScrollArea {
            background-color: #1e1e1e;
            border: none;
        }

        QScrollBar:vertical {
            background-color: #2b2b2b;
            width: 12px;
            border-radius: 6px;
        }

        QScrollBar::handle:vertical {
            background-color: #505050;
            border-radius: 6px;
            min-height: 20px;
        }

        QScrollBar::handle:vertical:hover {
            background-color: #606060;
        }

        QProgressBar {
            border: 2px solid #404040;
            border-radius: 6px;
            text-align: center;
            background-color: #2b2b2b;
            color: #e0e0e0;
            font-weight: 600;
        }

        QProgressBar::chunk {
            background-color: #00aaff;
            border-radius: 4px;
        }

        QStatusBar {
            background-color: #2b2b2b;
            border-top: 1px solid #404040;
            color: #e0e0e0;
        }

        QMenuBar {
            background-color: #2b2b2b;
            color: #e0e0e0;
            border-bottom: 1px solid #404040;
        }

        QMenuBar::item {
            background-color: transparent;
            padding: 6px 12px;
        }

        QMenuBar::item:selected {
            background-color: #404040;
            border-radius: 4px;
        }

        QMenu {
            background-color: #2b2b2b;
            color: #e0e0e0;
            border: 1px solid #404040;
            border-radius: 6px;
        }

        QMenu::item {
            padding: 8px 16px;
        }

        QMenu::item:selected {
            background-color: #404040;
        }

        QDialog {
            background-color: #1e1e1e;
        }
        """

        self.setStyleSheet(dark_style)

    def setup_ui(self):
        self.setWindowTitle("Graffiti - AI Image Tagging v1.1a")
        self.setGeometry(100, 100, 1400, 900)
        self.setMinimumSize(1000, 600)

        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        # Main layout
        main_layout = QVBoxLayout(central_widget)
        main_layout.setSpacing(15)
        main_layout.setContentsMargins(20, 20, 20, 20)

        # Top control panel
        control_panel = self.create_control_panel()
        main_layout.addWidget(control_panel)

        # Main content area with splitter
        splitter = QSplitter(Qt.Orientation.Horizontal)

        # Left panel - folder info and controls
        left_panel = self.create_left_panel()
        splitter.addWidget(left_panel)

        # Right panel - image grid
        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setMinimumWidth(900)

        self.image_grid_widget = QWidget()
        self.image_grid_layout = QGridLayout(self.image_grid_widget)
        self.image_grid_layout.setSpacing(15)
        self.scroll_area.setWidget(self.image_grid_widget)

        splitter.addWidget(self.scroll_area)
        splitter.setSizes([350, 1050])

        main_layout.addWidget(splitter)

        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        self.progress_bar.setMinimumHeight(25)
        main_layout.addWidget(self.progress_bar)

        # Status bar
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("Ready - Open a folder to begin")

    def create_control_panel(self):
        panel = QFrame()
        panel.setMinimumHeight(80)

        layout = QHBoxLayout(panel)
        layout.setSpacing(20)
        layout.setContentsMargins(25, 20, 25, 20)

        # Open Folder Button
        self.open_folder_btn = QPushButton("📁 Open Folder")
        self.open_folder_btn.clicked.connect(self.open_folder)
        self.open_folder_btn.setMinimumSize(150, 45)

        # Select/Deselect All
        self.select_all_btn = QPushButton("✓ Select All")
        self.select_all_btn.clicked.connect(self.select_all_images)
        self.select_all_btn.setEnabled(False)
        self.select_all_btn.setMinimumSize(120, 45)

        self.deselect_all_btn = QPushButton("✗ Deselect All")
        self.deselect_all_btn.clicked.connect(self.deselect_all_images)
        self.deselect_all_btn.setEnabled(False)
        self.deselect_all_btn.setMinimumSize(120, 45)

        # Process Button
        self.process_btn = QPushButton("🚀 Process Selected Images")
        self.process_btn.clicked.connect(self.process_images)
        self.process_btn.setEnabled(False)
        self.process_btn.setMinimumSize(200, 45)

        # Stop Button
        self.stop_btn = QPushButton("⏹ Stop Processing")
        self.stop_btn.clicked.connect(self.stop_processing)
        self.stop_btn.setEnabled(False)
        self.stop_btn.setVisible(False)
        self.stop_btn.setMinimumSize(150, 45)

        # Clear Memory Button
        self.clear_memory_btn = QPushButton("🧹 Clear GPU Memory")
        self.clear_memory_btn.clicked.connect(self.clear_ollama_memory)
        self.clear_memory_btn.setMinimumSize(150, 45)
        self.clear_memory_btn.setStyleSheet("""
            QPushButton {
                background-color: #d13212;
                color: white;
            }
            QPushButton:hover {
                background-color: #b02a0f;
            }
        """)

        layout.addWidget(self.open_folder_btn)
        layout.addStretch()
        layout.addWidget(self.select_all_btn)
        layout.addWidget(self.deselect_all_btn)
        layout.addStretch()
        layout.addWidget(self.clear_memory_btn)
        layout.addWidget(self.process_btn)
        layout.addWidget(self.stop_btn)

        return panel

    def create_left_panel(self):
        panel = QFrame()
        panel.setMaximumWidth(350)
        panel.setMinimumWidth(300)

        layout = QVBoxLayout(panel)
        layout.setSpacing(20)
        layout.setContentsMargins(20, 20, 20, 20)

        # Folder Info
        folder_group = QGroupBox("Current Folder")
        folder_layout = QVBoxLayout(folder_group)
        folder_layout.setSpacing(12)

        self.folder_label = QLabel("No folder selected")
        self.folder_label.setWordWrap(True)
        self.folder_label.setStyleSheet("font-size: 14px; color: #bbbbbb;")

        self.image_count_label = QLabel("Images: 0")
        self.image_count_label.setStyleSheet("font-size: 14px; font-weight: 600;")

        self.selected_count_label = QLabel("Selected: 0")
        self.selected_count_label.setStyleSheet("font-size: 14px; font-weight: 600; color: #00aaff;")

        folder_layout.addWidget(self.folder_label)
        folder_layout.addWidget(self.image_count_label)
        folder_layout.addWidget(self.selected_count_label)

        # LLM Status
        llm_group = QGroupBox("LLM Configuration")
        llm_layout = QVBoxLayout(llm_group)
        llm_layout.setSpacing(12)

        self.llm_status_label = QLabel(f"Provider: {self.settings.llm_provider}")
        self.llm_status_label.setStyleSheet("font-size: 14px; font-weight: 600;")

        self.llm_model_label = QLabel("Model: Not configured")
        self.llm_model_label.setStyleSheet("font-size: 14px; color: #bbbbbb;")

        llm_layout.addWidget(self.llm_status_label)
        llm_layout.addWidget(self.llm_model_label)

        # Processing Info
        process_group = QGroupBox("Processing Status")
        process_layout = QVBoxLayout(process_group)
        process_layout.setSpacing(12)

        self.process_status_label = QLabel("Status: Ready")
        self.process_status_label.setStyleSheet("font-size: 14px; font-weight: 600;")

        self.current_file_label = QLabel("Current: None")
        self.current_file_label.setStyleSheet("font-size: 12px; color: #bbbbbb;")
        self.current_file_label.setWordWrap(True)

        process_layout.addWidget(self.process_status_label)
        process_layout.addWidget(self.current_file_label)

        layout.addWidget(folder_group)
        layout.addWidget(llm_group)
        layout.addWidget(process_group)
        layout.addStretch()

        self.update_llm_status()

        return panel

    def setup_menu(self):
        menubar = self.menuBar()

        # File menu
        file_menu = menubar.addMenu("File")

        open_action = QAction("Open Folder", self)
        open_action.triggered.connect(self.open_folder)
        file_menu.addAction(open_action)

        file_menu.addSeparator()

        exit_action = QAction("Exit", self)
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)

        # Settings menu
        settings_menu = menubar.addMenu("Settings")

        config_action = QAction("Configure LLM", self)
        config_action.triggered.connect(self.open_settings)
        settings_menu.addAction(config_action)

        # Help menu
        help_menu = menubar.addMenu("Help")

        about_action = QAction("About", self)
        about_action.triggered.connect(self.show_about)
        help_menu.addAction(about_action)

    def open_folder(self):
        folder = QFileDialog.getExistingDirectory(
            self, "Select Image Folder", str(Path.home())
        )

        if folder:
            self.current_folder = folder
            self.load_images(folder)

    def load_images(self, folder_path: str):
        # Clear existing thumbnails
        self.clear_image_grid()

        # Supported image formats - fix duplicate issue
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.gif', '.webp'}

        # Find all images in the folder - use set to avoid duplicates
        image_files = set()
        folder = Path(folder_path)

        # Search for images with case-insensitive extension matching
        for file_path in folder.iterdir():
            if file_path.is_file():
                ext_lower = file_path.suffix.lower()
                if ext_lower in image_extensions:
                    image_files.add(file_path)

        # Create thumbnails
        self.image_thumbnails = []
        row, col = 0, 0
        max_cols = 5  # Increased for better layout

        for image_file in sorted(image_files):
            thumbnail = ImageThumbnail(str(image_file))
            thumbnail.checkbox.stateChanged.connect(self.update_selection_count)

            self.image_thumbnails.append(thumbnail)
            self.image_grid_layout.addWidget(thumbnail, row, col)

            col += 1
            if col >= max_cols:
                col = 0
                row += 1

        # Update UI
        self.folder_label.setText(f"📁 {os.path.basename(folder_path)}")
        self.image_count_label.setText(f"Images: {len(image_files)}")
        self.selected_count_label.setText("Selected: 0")

        # Enable controls
        self.select_all_btn.setEnabled(len(image_files) > 0)
        self.deselect_all_btn.setEnabled(len(image_files) > 0)

        self.status_bar.showMessage(f"Loaded {len(image_files)} images from {folder_path}")

    def clear_image_grid(self):
        while self.image_grid_layout.count():
            child = self.image_grid_layout.takeAt(0)
            if child.widget():
                child.widget().deleteLater()
        self.image_thumbnails.clear()

    def select_all_images(self):
        for thumbnail in self.image_thumbnails:
            thumbnail.checkbox.setChecked(True)

    def deselect_all_images(self):
        for thumbnail in self.image_thumbnails:
            thumbnail.checkbox.setChecked(False)

    def update_selection_count(self):
        selected_count = sum(1 for thumb in self.image_thumbnails if thumb.is_selected())
        self.selected_count_label.setText(f"Selected: {selected_count}")
        self.process_btn.setEnabled(selected_count > 0)

    def clear_ollama_memory(self):
        """Manually clear Ollama's GPU memory"""
        try:
            if self.settings.llm_provider == 'ollama':
                print("[UI] Manually clearing Ollama GPU memory...")

                unload_payload = {
                    "model": self.settings.ollama_model,
                    "keep_alive": 0
                }

                response = requests.post(
                    f"{self.settings.ollama_url}/api/generate",
                    json=unload_payload,
                    timeout=10
                )

                if response.ok:
                    self.status_bar.showMessage("✓ GPU memory cleared")
                    QMessageBox.information(self, "Memory Cleared",
                                            f"Successfully cleared GPU memory for {self.settings.ollama_model}")
                else:
                    self.status_bar.showMessage("✗ Failed to clear memory")
                    QMessageBox.warning(self, "Clear Failed",
                                        "Failed to clear GPU memory. Check if Ollama is running.")
            else:
                QMessageBox.information(self, "Not Applicable",
                                        f"Memory clearing only applies to Ollama. Current provider: {self.settings.llm_provider}")

        except Exception as e:
            self.status_bar.showMessage("✗ Error clearing memory")
            QMessageBox.critical(self, "Error", f"Error clearing memory:\n{str(e)}")

    def process_images(self):
        selected_images = [
            thumb.image_path for thumb in self.image_thumbnails
            if thumb.is_selected()
        ]

        if not selected_images:
            QMessageBox.warning(self, "Warning", "No images selected for processing.")
            return

        # Validate settings
        if not self.validate_llm_settings():
            return

        # Start processing
        self.progress_bar.setMaximum(len(selected_images))
        self.progress_bar.setValue(0)
        self.progress_bar.setVisible(True)

        # Update UI for processing state
        self.process_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.stop_btn.setVisible(True)
        self.process_status_label.setText("Status: Processing...")

        # Start processing with the manager
        self.processing_manager.start_processing(selected_images, self.settings)

    def stop_processing(self):
        """Stop processing immediately"""
        self.processing_manager.stop_processing()

        # Reset UI
        self.progress_bar.setVisible(False)
        self.process_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.stop_btn.setVisible(False)
        self.process_status_label.setText("Status: Stopped")
        self.current_file_label.setText("Current: None")
        self.status_bar.showMessage("Processing stopped by user")

    def validate_llm_settings(self):
        if self.settings.llm_provider == 'openai' and not self.settings.openai_api_key:
            QMessageBox.warning(self, "Configuration Error",
                                "OpenAI API key is required. Please configure in Settings.")
            return False
        return True

    @pyqtSlot(int, int, str)
    def update_progress(self, completed: int, total: int, current_filename: str):
        """Update progress display"""
        self.progress_bar.setValue(completed)
        self.current_file_label.setText(f"Current: {current_filename}")
        self.status_bar.showMessage(f"Processing {completed}/{total}: {current_filename}")

    @pyqtSlot(list)
    def processing_finished(self, completed_results: list):
        """Called when all processing is complete"""
        self.progress_bar.setVisible(False)
        self.process_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.stop_btn.setVisible(False)
        self.process_status_label.setText("Status: Complete ✓")
        self.current_file_label.setText("Current: None")

        # Show completion message
        message = f"Processing completed successfully!\n\n"
        message += f"Processed {len(completed_results)} images\n"
        message += f"Metadata has been written to the image files."

        self.status_bar.showMessage(f"✓ Completed processing {len(completed_results)} images")

        QMessageBox.information(self, "Processing Complete", message)

    @pyqtSlot(str)
    def processing_error(self, error: str):
        """Called when processing encounters an error"""
        self.progress_bar.setVisible(False)
        self.process_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.stop_btn.setVisible(False)
        self.process_status_label.setText("Status: Error ✗")
        self.current_file_label.setText("Current: None")
        self.status_bar.showMessage("Processing failed!")

        QMessageBox.critical(self, "Processing Error", f"Processing failed:\n\n{error}")

    def open_settings(self):
        dialog = SettingsDialog(self.settings, self)
        if dialog.exec() == QDialog.DialogCode.Accepted:
            self.update_llm_status()

    def update_llm_status(self):
        self.llm_status_label.setText(f"Provider: {self.settings.llm_provider}")
        if self.settings.llm_provider == 'ollama':
            self.llm_model_label.setText(f"Model: {self.settings.ollama_model}")
        elif self.settings.llm_provider == 'lmstudio':
            self.llm_model_label.setText(f"Model: {self.settings.lmstudio_model}")
        elif self.settings.llm_provider == 'openai':
            self.llm_model_label.setText("Model: GPT-4 Vision")

    def show_about(self):
        QMessageBox.about(self, "About Graffiti",
                          "Graffiti v1.1a 🚀\n\n"
                          "An AI-powered image tagging application\n"
                          "with intelligent sequential processing.\n\n"
                          "Author: Michael Wright\n\n"
                          "NEW in v1.1a:\n"
                          "• 🔍 Click thumbnails to zoom\n"
                          "• 🏷️ Improved hashtag parsing\n"
                          "• 🧹 Manual GPU memory clearing\n"
                          "• 📝 XMP-dc metadata for compatibility\n\n"
                          "Metadata Standards:\n"
                          "• XMP-dc:Description for descriptions\n"
                          "• XMP-dc:Subject for hashtags/keywords\n"
                          "• Compatible with Lightroom, Photoshop, etc.\n\n"
                          "Core Features:\n"
                          "• Single comprehensive prompts\n"
                          "• Non-blocking UI with retry logic\n"
                          "• Robust thread management\n"
                          "• Smart response parsing\n"
                          "• Automatic memory management\n\n"
                          "Built with PyQt6 and Python\n"
                          "Based on v1.0alpha (immortalized)\n\n"
                          "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
                          "© 2025 Michael Wright. All Rights Reserved.\n"
                          "This software is proprietary and confidential.\n"
                          "Unauthorized copying, distribution, or use is strictly prohibited.\n"
                          "This software is not free and requires proper licensing.")


def main():
    app = QApplication(sys.argv)
    app.setApplicationName("Graffiti")
    app.setOrganizationName("Graffiti")

    # Set application-wide font
    font = QFont("Segoe UI", 10)
    app.setFont(font)

    window = MainWindow()
    window.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()