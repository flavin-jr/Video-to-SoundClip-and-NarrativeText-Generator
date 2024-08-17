# Video-to-SoundClip and NarrativeText Generator

## Overview

This repository contains a Gradio-based application that processes YouTube videos or text inputs to generate custom soundtracks and narrative descriptions. The application uses advanced AI models to analyze video frames or text content and creates melodies and audio narratives that can be used as background music or voiceovers for videos.

## Features

- **Video-to-SoundClip**: 
  - Extracts frames from a YouTube video at a specified interval.
  - Uses AI models to describe the content of the video frames.
  - Generates a corresponding melody based on the video description.
  - Merges the generated audio with the original video to create a final video output with a custom soundtrack.

- **NarrativeText-to-Audio**:
  - Takes a text input and generates a melodic description using AI.
  - Converts the text into a narrative audio file.
  - Optionally merges the generated audio with a video.

## Dependencies

The application requires several Python libraries and tools:

- `os`, `shutil`, `subprocess`: For file manipulation and running shell commands.
- `gradio`: For creating the user interface.
- `pytube`: For downloading YouTube videos.
- `ffmpeg`: For audio and video processing.
- `torch`, `transformers`: For generating music from text using AI models.
- `pydub`: For manipulating audio files.
- `scipy`: For handling audio file writing.
- `colorama`: For colored terminal output (debugging purposes).
- `requests`: For making HTTP requests to external APIs.

## Installation

1. **Clone the repository**:
    ```sh
    git clone <repository-url>
    cd <repository-directory>
    ```

2. **Install the dependencies**:
    ```sh
    pip install -r requirements.txt
    ```

3. **Install FFmpeg**:
   - FFmpeg is required for processing audio and video files. You can download it from [FFmpeg's official website](https://ffmpeg.org/download.html) or install it via a package manager.

4. **Set up API keys**:
   - Replace the placeholder `"YOUR_OPENAI_API_KEY"` in the code with your OpenAI API key.
   - Replace the placeholder `"ELEVENLABS_API_KEY"` with your Eleven Labs API key.

## Usage

1. **Run the application**:
    ```sh
    python app.py
    ```

2. **Access the Gradio interface**:
   - Once the app is running, a URL will be provided in the terminal. Open it in your browser to interact with the app.

3. **Video-to-SoundClip**:
   - Paste a YouTube video link and choose the interval (in seconds) for frame extraction.
   - The app will process the video, generate a custom soundtrack, and merge it with the video.
   - Download the final video with the custom soundtrack.

4. **NarrativeText-to-Audio**:
   - Input a text description to generate a melody or narrative audio.
   - The generated audio can be downloaded or optionally merged with an uploaded video.

## How It Works

- **Video Processing**: 
  - The app uses `pytube` to download YouTube videos and `ffmpeg` to extract frames at specified intervals. 
  - The frames are analyzed by a CLIP model using the Hugging Face API, which provides textual descriptions of the frames.
  
- **Melody Generation**:
  - The app uses the `MusicgenForConditionalGeneration` model from Hugging Face to generate a melody based on the descriptions of the video frames.
  - The generated melody is merged with the original video using `ffmpeg`.

- **Text to Audio**:
  - The text input is processed using OpenAI's GPT-3.5 to generate a detailed melody description.
  - The description is converted into a melodic audio file using the Musicgen model.
  - If a video is provided, the audio is merged with it.

## Contributing

Contributions are welcome! Please feel free to submit a pull request or open an issue for any bugs or feature requests.


