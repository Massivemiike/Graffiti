# Image Hashtag Generator

A Python application with a GUI interface that utilizes ollama running locally in Docker to scan images, generate hashtags, and write metadata to images.

## Features

- Select and scan folders for images
- Choose which images to process
- Generate hashtags and descriptions using Vision Language Models via ollama
- Write metadata to images using exiftool
- Progress tracking and status updates
- Queuing system for processing multiple images

## Prerequisites

1. **Python 3.7+** installed on your system
2. **exiftool** installed:
   - **macOS**: Install via Homebrew: `brew install exiftool`
   - **Windows**: Download from [ExifTool website](https://exiftool.org/) and add to your PATH
   - **Linux**: Install via package manager, e.g., `sudo apt install libimage-exiftool-perl`

3. **ollama** running in Docker:
   ```bash
   docker run -d --gpus=all -v ollama:/root/.ollama -p 11434:11434 --name ollama ollama/ollama
   ```

4. **Vision Language Models** pulled in ollama:
   ```bash
   docker exec -it ollama ollama pull llava
   ```
   You can also pull other models like `bakllava`, `llava-13b`, or `llava-34b`.

## Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/yourusername/image-hashtag-generator.git
   cd image-hashtag-generator
   ```

2. Install the required Python packages:
   ```bash
   pip install -r requirements.txt
   ```

3. Verify your setup:
   ```bash
   python test_setup.py
   ```
   This script will check if all required dependencies are installed and properly configured.

## Usage

1. Run the application:
   ```bash
   python main.py
   ```

2. Select an ollama model from the dropdown menu.

3. Click "Load Folder" to select a folder containing images.

4. Select the images you want to process from the list (use Ctrl/Cmd+click for multiple selection).

5. Click "Process Selected Images" to start processing.

6. The progress bar will show the current progress, and a success message will appear when all images have been processed.

## How It Works

1. The application scans the selected folder for image files.
2. Selected images are added to a processing queue.
3. Each image is processed one at a time:
   - The image is sent to the ollama API running locally in Docker.
   - ollama generates 10 hashtags and a short paragraph description.
   - The hashtags and description are written to the image's IPTC metadata using exiftool.
4. Progress is displayed in the status bar.

## Troubleshooting

- **ExifTool not found**: Make sure exiftool is installed and in your system PATH.
- **Connection error to ollama**: Ensure the ollama Docker container is running on port 11434.
- **Model not found**: Make sure you've pulled the selected model in ollama.
- **Image processing errors**: Check that the images are valid and not corrupted.

## License

MIT
