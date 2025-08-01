# 🎨 Graffiti App - AI-Powered Image Tagging Tool

<div align="center">

![Python](https://img.shields.io/badge/python-v3.8+-blue.svg)
![PyQt6](https://img.shields.io/badge/PyQt6-GUI-green.svg)
![License](https://img.shields.io/badge/license-MIT-blue.svg)
![AI Powered](https://img.shields.io/badge/AI-Powered-purple.svg)

*Transform your image collection with intelligent AI-generated descriptions and hashtags*

[Features](#-features) • [Installation](#-installation) • [Usage](#-usage) • [Configuration](#-configuration) • [Contributing](#-contributing)

</div>

---

## 📖 About

**Graffiti App** is a powerful, user-friendly desktop application that leverages cutting-edge AI technology to automatically generate comprehensive descriptions and relevant hashtags for your images. Whether you're a content creator, social media manager, or just someone looking to organize their photo collection, Graffiti App streamlines the process of image tagging with support for multiple AI providers.

### 🎯 Why Graffiti App?

- **Multi-AI Support**: Works seamlessly with Ollama, LMStudio, and OpenAI API
- **Batch Processing**: Handle hundreds of images efficiently
- **Smart Analysis**: Generate detailed descriptions and contextual hashtags
- **User-Friendly Interface**: Intuitive PyQt6-based GUI with dark theme
- **Flexible Configuration**: Customize AI models and processing parameters
- **Metadata Integration**: Automatically saves tags and descriptions to image metadata

---

## ✨ Features

### 🖼️ **Image Processing**
- **Batch Processing**: Select and process multiple images simultaneously
- **Smart Thumbnails**: Preview images with zoom functionality
- **Progress Tracking**: Real-time processing status and progress bars
- **Error Handling**: Robust error management with retry mechanisms

### 🤖 **AI Integration**
- **Multiple Providers**: 
  - 🦙 **Ollama** - Local AI models (llava, bakllava, etc.)
  - 🏠 **LMStudio** - Local model hosting
  - 🌐 **OpenAI** - GPT-4 Vision API
- **Comprehensive Analysis**: Detailed scene descriptions, object detection, and context understanding
- **Smart Hashtag Generation**: Relevant, trending hashtags based on image content

### 🎨 **User Interface**
- **Modern Dark Theme**: Easy on the eyes with professional aesthetics
- **Intuitive Layout**: Clean, organized interface with logical workflow
- **Image Grid View**: Visual thumbnail grid with selection capabilities
- **Settings Management**: Persistent configuration with easy access

### ⚙️ **Advanced Features**
- **Memory Management**: Automatic Ollama memory clearing for optimal performance
- **Concurrent Processing**: Multi-threaded processing for faster results
- **Health Monitoring**: AI service health checks and status indicators
- **Metadata Writing**: Automatic saving of descriptions and hashtags to image files

---

## 🚀 Installation

### Prerequisites

- **Python 3.8+** installed on your system
- **PyQt6** for the graphical interface
- At least one AI provider configured (Ollama, LMStudio, or OpenAI API)

### Step 1: Clone the Repository

```bash
git clone https://github.com/yourusername/graffiti-app.git
cd graffiti-app
```

### Step 2: Install Dependencies

```bash
pip install PyQt6 requests
```

### Step 3: Set Up AI Provider

Choose one or more of the following options:

#### Option A: Ollama (Recommended for Local Processing)
1. Install Ollama from [ollama.ai](https://ollama.ai)
2. Pull a vision model:
   ```bash
   ollama pull llava
   # or
   ollama pull bakllava
   ```

#### Option B: LMStudio
1. Download and install [LMStudio](https://lmstudio.ai)
2. Load a vision-capable model
3. Start the local server

#### Option C: OpenAI API
1. Get your API key from [OpenAI](https://platform.openai.com)
2. Configure it in the app settings

---

## 📱 Usage

### Quick Start

1. **Launch the Application**
   ```bash
   python graffiti_app.py
   ```

2. **Configure Settings**
   - Click the ⚙️ Settings button
   - Select your AI provider (Ollama/LMStudio/OpenAI)
   - Configure model parameters and API settings

3. **Load Images**
   - Click "📁 Open Folder" to select your image directory
   - Images will appear as thumbnails in the grid

4. **Select and Process**
   - Use checkboxes to select images for processing
   - Click "🚀 Process Selected Images"
   - Monitor progress in the status bar

### Detailed Workflow

#### 🖼️ **Image Selection**
- **Individual Selection**: Click checkboxes on specific images
- **Select All**: Use the "Select All" button for batch operations
- **Preview**: Click on any thumbnail to view full-size image

#### ⚙️ **Configuration Options**

| Setting | Description | Default |
|---------|-------------|---------|
| **AI Provider** | Choose between Ollama, LMStudio, or OpenAI | Ollama |
| **Model** | Specific model to use (e.g., llava, gpt-4-vision) | llava |
| **Server URL** | Local server address for Ollama/LMStudio | http://localhost:11434 |
| **API Key** | OpenAI API key (if using OpenAI) | - |
| **Timeout** | Request timeout in seconds | 120 |
| **Max Retries** | Number of retry attempts | 3 |

#### 🔄 **Processing Flow**
1. **Image Analysis**: AI analyzes visual content, objects, scenes, and context
2. **Description Generation**: Creates detailed, natural language descriptions
3. **Hashtag Creation**: Generates relevant hashtags based on content
4. **Metadata Writing**: Saves results to image metadata
5. **Progress Updates**: Real-time status updates and completion notifications

---

## 🔧 Configuration

### AI Provider Setup

#### Ollama Configuration
```json
{
  "provider": "ollama",
  "model": "llava",
  "server_url": "http://localhost:11434",
  "timeout": 120
}
```

#### LMStudio Configuration
```json
{
  "provider": "lmstudio",
  "model": "your-vision-model",
  "server_url": "http://localhost:1234",
  "timeout": 120
}
```

#### OpenAI Configuration
```json
{
  "provider": "openai",
  "model": "gpt-4-vision-preview",
  "api_key": "your-api-key-here",
  "timeout": 60
}
```

### Advanced Settings

- **Memory Management**: Automatic Ollama memory clearing between batches
- **Concurrent Processing**: Configurable thread pool for parallel processing
- **Error Handling**: Automatic retry with exponential backoff
- **Health Monitoring**: Service availability checks before processing

---

## 🎨 Screenshots

*Coming soon - Screenshots will be added to showcase the beautiful interface and workflow*

---

## 🤝 Contributing

We welcome contributions! Here's how you can help:

1. **Fork the repository**
2. **Create a feature branch** (`git checkout -b feature/amazing-feature`)
3. **Commit your changes** (`git commit -m 'Add amazing feature'`)
4. **Push to the branch** (`git push origin feature/amazing-feature`)
5. **Open a Pull Request**

### Development Guidelines
- Follow PEP 8 style guidelines
- Add docstrings to new functions and classes
- Test your changes thoroughly
- Update documentation as needed

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🚀 Future Improvements

### 🎯 **Planned Features**

#### **Enhanced AI Capabilities**
- [ ] **Multi-Model Ensemble**: Combine results from multiple AI models for improved accuracy
- [ ] **Custom Model Training**: Support for fine-tuned models specific to user domains
- [ ] **Batch Optimization**: Smart batching algorithms for improved processing speed
- [ ] **Context Awareness**: Remember previous images for better contextual understanding

#### **Advanced Image Processing**
- [ ] **Image Enhancement**: Pre-processing filters to improve AI analysis quality
- [ ] **Format Support**: Extended support for RAW, HEIC, and other image formats
- [ ] **Duplicate Detection**: Identify and handle duplicate or similar images
- [ ] **Quality Assessment**: Automatic image quality scoring and filtering

#### **User Experience Improvements**
- [ ] **Drag & Drop Interface**: Direct file dropping for easier workflow
- [ ] **Keyboard Shortcuts**: Comprehensive hotkey support for power users
- [ ] **Custom Themes**: Multiple UI themes and customization options
- [ ] **Workspace Management**: Save and load different project configurations

#### **Export & Integration**
- [ ] **Export Formats**: CSV, JSON, XML export options for metadata
- [ ] **Cloud Integration**: Direct upload to social media platforms
- [ ] **Database Support**: SQLite/PostgreSQL integration for large collections
- [ ] **API Endpoints**: RESTful API for integration with other tools

#### **Performance & Scalability**
- [ ] **GPU Acceleration**: CUDA support for faster local processing
- [ ] **Distributed Processing**: Multi-machine processing capabilities
- [ ] **Caching System**: Intelligent result caching to avoid reprocessing
- [ ] **Memory Optimization**: Improved memory usage for large image sets

#### **Analytics & Insights**
- [ ] **Processing Statistics**: Detailed analytics on processing performance
- [ ] **Tag Analytics**: Insights into most common tags and descriptions
- [ ] **Quality Metrics**: Accuracy scoring and improvement suggestions
- [ ] **Usage Reports**: Comprehensive usage statistics and trends

#### **Enterprise Features**
- [ ] **User Management**: Multi-user support with role-based access
- [ ] **Audit Logging**: Comprehensive logging for enterprise compliance
- [ ] **Batch Scheduling**: Automated processing schedules
- [ ] **Integration APIs**: Enterprise system integration capabilities

---

<div align="center">

### 🌟 Star this repository if you find it useful!

**Made with ❤️ by the Graffiti App Team**

[Report Bug](https://github.com/yourusername/graffiti-app/issues) • [Request Feature](https://github.com/yourusername/graffiti-app/issues) • [Documentation](https://github.com/yourusername/graffiti-app/wiki)

</div>