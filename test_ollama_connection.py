import sys

import requests


def test_ollama_connection():
    """Test connection to Ollama server."""
    print("Testing connection to Ollama server at http://100.124.224.80:11434...")
    
    try:
        # Test basic connection
        response = requests.get("http://100.124.224.80:11434/api/tags")
        if response.status_code == 200:
            print("✅ Successfully connected to Ollama server!")
            models = response.json().get("models", [])
            
            if models:
                print(f"\nAvailable models:")
                for model in models:
                    print(f"- {model['name']}")
            else:
                print("\n⚠️ No models found. You need to pull at least one vision model.")
                print("Try running: docker exec -it ollama ollama pull llava")
            
            # Test if any vision models are available
            vision_models = ["llava", "bakllava", "llava-13b", "llava-34b"]
            available_vision_models = [model['name'] for model in models if model['name'] in vision_models]
            
            if available_vision_models:
                print(f"\n✅ Found vision models: {', '.join(available_vision_models)}")
            else:
                print("\n⚠️ No vision models found. The application requires at least one vision model.")
                print("Try running: docker exec -it ollama ollama pull llava")
            
            return True
        else:
            print(f"❌ Connection failed with status code: {response.status_code}")
            return False
    
    except requests.exceptions.ConnectionError:
        print("❌ Connection error: Could not connect to Ollama server at http://100.124.224.80:11434")
        print("\nPossible solutions:")
        print("1. Make sure Ollama is running in Docker:")
        print("   docker run -d --name ollama -p 11434:11434 ollama/ollama")
        print("2. Check if the container is running:")
        print("   docker ps | grep ollama")
        return False
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        return False

if __name__ == "__main__":
    success = test_ollama_connection()
    sys.exit(0 if success else 1)