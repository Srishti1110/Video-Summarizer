---

# Video Summarization with Whisper & Groq API

This project allows you to upload a video, extract its audio, transcribe the audio to text, and generate a summary of the transcription. It uses the Whisper model for transcription and Groq's API for summarization. The summary is then displayed in a Gradio interface, and an SRT file is generated for video subtitles.

## Table of Contents
1. [Overview](#overview)
2. [Setup](#setup)
3. [Project Flow](#project-flow)
4. [Use Cases](#use-cases)
5. [Features](#features)
6. [Directory Structure](#directory-structure)
7. [Dependencies](#dependencies)
8. [How to Use](#how-to-use)
9. [Logging and Debugging](#logging-and-debugging)
10. [License](#license)

---

## Overview

This project automates the process of video summarization. Here's how it works:
1. **Video Upload:** You upload a video file via the Gradio interface.
2. **Audio Extraction:** The video’s audio is extracted using `pydub`.
3. **Transcription:** The audio is then transcribed using the Whisper model.
4. **Summarization:** The transcription is processed, and a summary is generated using Groq's AI API.
5. **SRT Generation:** An SRT file is created for subtitles.
6. **Output:** You can download the SRT file and view the summary.

This is a powerful tool for creating video summaries, enhancing accessibility with subtitles, and summarizing long lectures, presentations, or meetings.

---

## Setup

### Prerequisites

Before setting up the project, ensure you have the following installed:
- **Python 3.10**
- **FFmpeg** (for video and audio processing)
- **Pip** (Python package installer)

### Steps

1. **Clone the repository:**

    ```bash
    git clone https://github.com/your-username/video-summarization.git
    cd video-summarization
    ```

2. **Create and activate a virtual environment (optional but recommended):**

    ```bash
    python -m venv .videosummarizer
    source .videosummarizer/bin/activate  # On Windows use: ./.videosummarizer/Scripts/activate
    ```

3. **Install the required dependencies:**

    ```bash
    pip install -r requirements.txt
    ```

4. **Set up environment variables:**

    Create a `.env` file in the root directory and add the following:

    ```
    GROQ_API_KEY="<your_api_key_here>"
    GROQ_MODEL_NAME="meta-llama/llama-4-scout-17b-16e-instruct"
    ```

    - Get your **Groq API Key** from [Groq](https://groq.com).
    - The default `GROQ_MODEL_NAME` is `gemma2-9b-it`, but you can change it to any Groq-supported model if needed.
    - Recommended model is `meta-llama/llama-4-scout-17b-16e-instruct`

5. **Install FFmpeg:**

    FFmpeg is required for audio extraction. Follow the installation guide from [FFmpeg](https://ffmpeg.org/download.html).

---

## Project Flow

1. **Video Upload:** The user uploads a `.mp4` video file.
2. **Audio Extraction:** The audio is extracted from the video using `pydub`.
3. **Audio Conversion:** The audio is converted to a tensor using `torch` and `numpy`.
4. **Transcription:** The audio is transcribed using the Whisper model.
5. **Summarization:** The transcription is summarized using Groq's AI model.
6. **SRT Generation:** An SRT file is created, which can be downloaded by the user.
7. **Output:** A summary of the video is displayed, and the SRT file is made available for download.

---

## Use Cases

- **Education:** Summarizing educational lectures or tutorials.
- **Meetings:** Summarizing business meetings or conferences.
- **Content Creation:** Creating summaries for podcasts, interviews, or video blogs.
- **Accessibility:** Generating subtitles for videos to improve accessibility.

---

## Features

- **Gradio Interface:** Easy-to-use web interface for video uploading and output display.
- **Whisper Integration:** Uses Whisper's speech-to-text model for accurate transcription.
- **Groq AI Integration:** Groq's AI model is used to summarize the transcription.
- **SRT Generation:** Automatically creates an SRT subtitle file for the video.
- **Asynchronous Processing:** Efficient asynchronous operations for video processing, transcription, and summarization.
- **Memory Profiling:** Logs memory usage to monitor resource consumption during processing.

---

## Directory Structure

```bash
video-summarization/
│
├── uploads/              # Temporary directory for uploaded video files
├── output/               # Directory for saving processed audio, summaries, and SRT files
├── .env                  # Environment variables (API keys, configuration)
├── requirements.txt      # List of dependencies
├── summarizer_log.log    # Logs of the processing steps
├── app.py                # Main Python script
└── README.md             # This README file
```

---

## Dependencies

- **torch**: For tensor operations and model loading.
- **gradio**: For the web interface.
- **whisper**: For audio transcription.
- **pydub**: For audio extraction from video.
- **soundfile**: For audio file processing.
- **dotenv**: For managing environment variables.
- **langchain**: For AI-powered text summarization (Groq API).
- **asyncio**: For asynchronous file handling and operations.

Install dependencies using:

```bash
pip install -r requirements.txt
```

---

## How to Use

1. **Start the Gradio Interface:**

   Run the following command to start the Gradio interface:

   ```bash
   gradio app.py
   ```

   The interface will open in your web browser.

2. **Upload a Video:**
   - Click on the file uploader and select an `.mp4` video file from your computer.

3. **Process the Video:**
   - The app will process the video, extract audio, transcribe it, and summarize the transcription.
   - The status of each step will be displayed in the Gradio interface.

4. **Download the SRT File:**
   - After processing, you can download the generated SRT subtitle file.

5. **View the Summary:**
   - The summary of the video will be displayed in the text box.

---

## Logging and Debugging

The project logs important events and errors to the `summarizer_log.log` file. You can check the log file to see:

- **Audio extraction and conversion details.**
- **Transcription process and any errors.**
- **Summarization steps and issues.**
- **Memory usage during each stage.**

The log is configured to show timestamps in IST (Indian Standard Time).

---

## License

This project is licensed under the MIT License.

---
