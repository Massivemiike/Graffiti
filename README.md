# Image Tagger with Ollama

A Python application with a GUI interface that uses Ollama's vision language models to generate hashtags and descriptions for images, then writes this metadata to the images using ExifTool.

## Features

- Load a folder of images and select which ones to process
- Process images through Ollama's vision language models
- Generate 10 hashtags and a short description for each image
- Write metadata to images using ExifTool
- Queue system for processing images one at a time
- Progress bar to track processing status

## Prerequisites

1. **Python 3.6+** with the following packages:
   - PyQt5
   - requests

2. **Ollama** running in Docker:
   ```bash
   docker run -d --name ollama -p 11434:11434 ollama/ollama
   ```

3. **ExifTool** installed:
   - On macOS: `brew install exiftool`
   - On Windows: Download from [ExifTool website](https://exiftool.org/) and add to your PATH

4. **Vision Language Models** pulled in Ollama:
   ```bash
   docker exec -it ollama ollama pull llava
   ```
   You can also pull other models like bakllava, llava-13b, or llava-34b.

## Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/yourusername/image-tagger.git
   cd image-tagger
   ```

2. Install the required Python packages:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

1. Start the application:
   ```bash
   python main.py
   ```

2. Click "Load Folder" to select a folder containing images.

3. Select the images you want to process from the list.

4. Choose a vision language model from the dropdown menu.

5. Click "Process Selected Images" to start processing.

6. The progress bar will show the current progress, and a message will appear when processing is complete.

## How It Works

1. The application scans the selected folder for image files (jpg, jpeg, png, gif, bmp, tiff, webp).

2. When you select images and click "Process", each image is:
   - Encoded as base64
   - Sent to Ollama's API with a prompt to generate hashtags and a description
   - The response is parsed to extract hashtags and description
   - ExifTool writes this metadata to the image's IPTC fields

3. The queuing system ensures that images are processed one at a time to avoid overwhelming the Ollama service.

## Testing Your Setup

Before running the main application, you can verify that your setup is working correctly:

### 1. Test Ollama Connection

```bash
python test_ollama_connection.py
```

This script will:
- Test the connection to Ollama
- List all available models
- Check if any vision models are available
- Provide troubleshooting tips if needed

### 2. Test ExifTool Installation

```bash
python test_exiftool.py
```

This script will:
- Check if ExifTool is installed and accessible
- Display the ExifTool version
- Provide installation instructions if ExifTool is not found

## Troubleshooting

- **ExifTool not found**: Run the ExifTool test script to verify your installation:
  ```bash
  python test_exiftool.py
  ```
  The script will provide installation instructions if ExifTool is not found. On macOS, ExifTool should be in `/usr/local/bin/exiftool` or `/opt/homebrew/bin/exiftool`.

- **Ollama connection error**: Ensure Ollama is running in Docker and accessible at http://localhost:11434. You can test with:
  ```bash
  python test_ollama_connection.py
  ```
  or
  ```bash
  curl http://localhost:11434/api/tags
  ```

- **Model not found**: Make sure you've pulled the model you're trying to use:
  ```bash
  docker exec -it ollama ollama pull <model-name>
  ```

## License

MIT
