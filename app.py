# Standard library imports
import os
import sys
import logging
import time
from datetime import timedelta, datetime
from pytz import timezone
from pathlib import Path
from typing import List, Tuple, Optional, Any, Dict
import asyncio
import aiofiles
import tracemalloc

# Third-party imports
import torch
import shutil
import numpy as np
import soundfile as sf
import gradio as gr
from pydub import AudioSegment
import whisper
from dotenv import load_dotenv

# Local or project-specific imports
from langchain_groq import ChatGroq
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.messages import SystemMessage


# Set event loop policy for Windows platform
if sys.platform.startswith("win"):
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

# Start memory tracking
tracemalloc.start()

# Set up logging with IST timezone
class ISTFormatter(logging.Formatter):
    def formatTime(self, record, datefmt=None):
        ist_offset = timedelta(hours=5, minutes=30)
        ist = timezone(ist_offset)
        record_time = datetime.fromtimestamp(record.created, tz=ist)
        return record_time.strftime(datefmt or "%Y-%m-%d %H:%M:%S")

# Configure logging formatter and file handler
formatter = ISTFormatter(fmt='%(asctime)s - %(levelname)s - %(message)s')
file_handler = logging.FileHandler("summarizer_log.log")
file_handler.setFormatter(formatter)

# Set up logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.addHandler(file_handler)

# Load environment variables
load_dotenv()

GROQ_MODEL_NAME = os.getenv("GROQ_MODEL_NAME", "gemma2-9b-it")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY:
    logging.error("GROQ_API_KEY must be set in the environment variables.")
    sys.exit(1)

# ------------------ CONFIGURATION ------------------

AudioSegment.ffprobe = r"C:/Program Files/ffmpeg/bin/ffprobe.exe"

# Modular configuration dictionary
CONFIG: Dict[str, Any] = {
    "BASE_DIR": os.getcwd(),
    "UPLOADS_DIR": os.path.join(os.getcwd(), "uploads"),
    "OUTPUT_DIR": os.path.join(os.getcwd(), "output"),
    "OUTPUT_WAV": os.path.join(os.getcwd(), "output", "audio_output.mp3"),
    "VIDEO_INPUT": os.path.join(os.getcwd(), "uploads", "input.mp4"),
    "WHISPER_MODEL_NAME": "base",  # default whisper model name
}

# Ensure necessary directories exist
for key in ["UPLOADS_DIR", "OUTPUT_DIR"]:
    os.makedirs(CONFIG[key], exist_ok=True)

# Global cache for Whisper model
WHISPER_MODEL_CACHE: Dict[str, Any] = {}

# LLM for summary generation
llm = ChatGroq(model=GROQ_MODEL_NAME, api_key=GROQ_API_KEY, temperature=0)

# ------------------ MEMORY PROFILING UTILITY ------------------

def log_memory_usage(stage: str = "") -> None:
    """Log current and peak memory usage."""
    current, peak = tracemalloc.get_traced_memory()
    logging.info(f"[Memory] {stage} - Current: {current/1024:.2f} KB, Peak: {peak/1024:.2f} KB")

# ------------------ ASYNCHRONOUS UTILITY FUNCTIONS ------------------

async def clear_folder(folder_path: str) -> None:
    """
    Asynchronously remove all files and folders in the given directory.
    Raises:
        OSError: If file removal fails.
    """
    if os.path.exists(folder_path) and os.path.isdir(folder_path):
        for file_name in os.listdir(folder_path):
            file_path = os.path.join(folder_path, file_name)
            try:
                if os.path.isfile(file_path):
                    await asyncio.to_thread(os.remove, file_path)
                else:
                    await asyncio.to_thread(shutil.rmtree, file_path)
            except (OSError, FileNotFoundError) as e:
                logging.error(f"Failed to delete {file_path}: {e}")

async def extract_audio_from_video(video_path: str) -> Optional[AudioSegment]:
    """
    Asynchronously extract audio from the given video file.
    Args:
        video_path (str): Path to the video file.
    Returns:
        Optional[AudioSegment]: Extracted audio or None if extraction fails.
    """
    try:
        if video_path is not None:
            audio = await asyncio.to_thread(AudioSegment.from_file, video_path)
            logging.info("Audio extraction complete.")
            return audio

    except (OSError, Exception) as e:
        logging.error(f"Error extracting audio: {e}")
        return None

async def audiosegment_to_tensor(audio_segment: AudioSegment) -> Tuple[Optional[torch.Tensor], Optional[int]]:
    """
    Convert AudioSegment to a normalized torch tensor and return sample rate.
    """
    try:
        np_audio = np.array(audio_segment.get_array_of_samples())
        if audio_segment.channels == 2:
            np_audio = np_audio.reshape((-1, audio_segment.channels)).T
        else:
            np_audio = np_audio.reshape((1, -1))
        
        max_val = float(1 << (8 * audio_segment.sample_width - 1))
        tensor_audio = torch.tensor(np_audio, dtype=torch.float32) / max_val
        return tensor_audio, audio_segment.frame_rate
    except Exception as e:
        logging.error(f"Error converting audio to tensor: {e}")
        return None, None

async def save_audio(waveform: torch.Tensor, sample_rate: int, output_path: str) -> None:
    """
    Save the waveform tensor as an MP3 file asynchronously at 192kbps.
    """
    try:
        np_audio = (waveform.numpy().T * (1 << 15)).astype(np.int16)  # Convert back to int16 PCM
        audio_segment = AudioSegment(
            np_audio.tobytes(),
            frame_rate=sample_rate,
            sample_width=2,  # 16-bit PCM (2 bytes per sample)
            channels=1 if len(np_audio.shape) == 1 else np_audio.shape[1]
        )
        await asyncio.to_thread(audio_segment.export, output_path, format="mp3", bitrate="192k")
        logging.info(f"Saved audio to {output_path} as MP3 (192kbps)")
    except Exception as e:
        logging.error(f"Error saving audio: {e}")

def format_timestamp(seconds: float) -> str:
    milliseconds = int((seconds - int(seconds)) * 1000)
    td = timedelta(seconds=int(seconds))
    return f"{str(td)}.{milliseconds:03}".replace(",", ".")

async def create_srt(transcription: dict, srt_output_path: str) -> bool:
    """
    Populate the vectorstore asynchronously using transcription data and write an SRT file.
    """
    try:
        async def write_srt() -> None:
            try:
                async with aiofiles.open(srt_output_path, "w", encoding="utf-8") as f:
                    for idx, segment in enumerate(transcription.get("segments", []), start=1):
                        start_time = format_timestamp(segment["start"])
                        end_time = format_timestamp(segment["end"])
                        text = segment["text"].strip()
                        await f.write(f"{idx}\n{start_time} --> {end_time}\n{text}\n\n")
            except Exception as e:
                logging.error(f"Error writing SRT file: {e}")
                raise

        await write_srt()
        logging.info(f"SRT file generated at {srt_output_path}")

        return True
    except Exception as e:
        logging.error(f"Error populating vectorstore: {e}")
        return False

async def transcribe_audio(input_audio_path: str, srt_output_path: str, model_name: str = CONFIG["WHISPER_MODEL_NAME"]) -> None:
    """
    Transcribe audio using Whisper and populate the vectorstore with results asynchronously.
    Utilizes model caching to avoid reloading the model for multiple transcriptions.
    """
    global WHISPER_MODEL_CACHE
    try:
        if model_name in WHISPER_MODEL_CACHE:
            model = WHISPER_MODEL_CACHE[model_name]
            logging.info(f"Using cached Whisper model: {model_name}")
        else:
            model = whisper.load_model(model_name)
            WHISPER_MODEL_CACHE[model_name] = model
            logging.info(f"Loaded and cached Whisper model: {model_name}")
        
        result = await asyncio.to_thread(model.transcribe, input_audio_path)
        logging.info("Transcription complete.")
        log_memory_usage("After transcription")
        
        if await create_srt(result, srt_output_path):
            logging.info(f"Transcription and vectorstore update successful; SRT saved at {srt_output_path}")
        
        summary = await get_summary(result["text"])
        
        if summary:
            logging.info("Summarization complete.")
            return summary
        else:
            logging.error("Error: Summarization failed.")
            return "Error: Summarization failed."
        
    except (RuntimeError, OSError) as e:
        logging.error(f"Error transcribing audio: {e}")

async def save_video_file(source_path: str, dest_directory: str, dest_filename: str) -> None:
    """
    Asynchronously save the uploaded video file to the designated directory.
    """
    os.makedirs(dest_directory, exist_ok=True)
    dest_path = os.path.join(dest_directory, dest_filename)
    try:
        async with aiofiles.open(source_path, 'rb') as src_file:
            data = await src_file.read()
        async with aiofiles.open(dest_path, 'wb') as dest_file:
            await dest_file.write(data)
        logging.info(f"Video file saved to {dest_path}")
    except (OSError, Exception) as e:
        logging.error(f"Error saving video file: {e}")

# ------------------ VIDEO PROCESSING & QUERY ------------------

async def gradio_video_upload(video_file: str) -> Any:
    summary = ""
    """
    Gradio callback to process an uploaded video and yield status updates.
    """
    yield "Starting video processing...", None, summary
    
    yield "Saving video file...", None, summary
    await save_video_file(source_path=video_file, dest_directory=CONFIG["UPLOADS_DIR"], dest_filename="input.mp4")
    yield "Video file saved.", None, summary
    
    yield "Extracting audio from video...", None, summary
    audio_segment = await extract_audio_from_video(video_file)
    if audio_segment is None:
        return
    
    yield "Converting audio to tensor...", None, summary
    waveform, sample_rate = await audiosegment_to_tensor(audio_segment)
    if waveform is None or sample_rate is None:
        yield "Error: Audio conversion failed.", None, summary
        return

    logging.info(f"Audio shape: {waveform.shape}, Sample rate: {sample_rate}")
    log_memory_usage("After audio conversion")
    
    yield "Saving audio...", None, "Summary will be generated after transcription."
    await save_audio(waveform, sample_rate, CONFIG["OUTPUT_WAV"])
    
    yield "Transcribing and summarizing...", None, "Hold on, this may take a while."
    srt_output = os.path.join(CONFIG["OUTPUT_DIR"], f"{Path(video_file).stem}.srt")
    summary = await transcribe_audio(CONFIG["OUTPUT_WAV"], srt_output, model_name=CONFIG["WHISPER_MODEL_NAME"])
    
    yield "Processing complete!", srt_output, summary
    log_memory_usage("After processing video")

async def get_summary(transcription: str, chunk_size: int = 4096, chunk_overlap: int = 256) -> str:
    """
    Hybrid summarization optimized for video transcription.
    """
    try:
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
        chunks = text_splitter.split_text(transcription)

        current_summary = ""

        for chunk in chunks:
            rolling_prompt = f"""
                ROLE: Summarizer for spoken video transcripts.
                TASK: Update the current summary with key points from the new chunk.
                RULES: 
                - Keep it concise, coherent, and non-redundant.
                - Ignore filler words and irrelevant details.

                CURRENT SUMMARY:
                <summary>{current_summary}</summary>

                NEW CHUNK:
                <chunk>{chunk}</chunk>

                OUTPUT:
                <updated_summary>
                """.strip()
            response = await llm.ainvoke([SystemMessage(content=rolling_prompt)])
            current_summary = response.content.strip()

            await asyncio.sleep(3)

        compression_prompt = f"""
            ROLE: Final editor.
            TASK: Polish the raw summary.
            RULES:
            - Improve clarity, flow, and engagement.
            - Remove redundancy and informal phrasing.
            
            RAW SUMMARY:
            <raw_summary>{current_summary}</raw_summary>
            
            OUTPUT:
            
            Short summary of the video:

            This video is about <walkthrough_summary>

            Detailed summary of the video:
            
            <detailed_summary>

            """.strip()
        final_response = await llm.ainvoke([SystemMessage(content=compression_prompt)])
        final_summary = final_response.content.strip()

        return final_summary

    except Exception as e:
        logging.error(f"Error during hybrid summarization: {e}")
        return "Error: Summarization failed."

# ------------------ CLEANUP FUNCTION ------------------

async def cleanup_temp_files() -> None:
    await asyncio.gather(
        clear_folder(CONFIG["UPLOADS_DIR"]),
        clear_folder(CONFIG["OUTPUT_DIR"])
    )
    logging.info("Temporary files cleaned up.")

# ------------------ GRADIO INTERFACE ------------------

with gr.Blocks(theme='davehornik/Tealy') as demo:
    with gr.Tab("Video Summarization"):
        with gr.Row():
            # First column: upload and status
            with gr.Column(scale=1, min_width=50):
                video_uploader = gr.File(
                    label="Upload Video",
                    type="filepath",
                    file_types=[".mp4"]
                )
                upload_status = gr.Textbox(label="Status", interactive=False)
                srt_download = gr.File(label="Download SRT")
            
            # Second column: summary output with markdown support
            with gr.Column(scale=1, min_width=50):
                summary_output = gr.Markdown(label="Summary")

        # Link the function to the components
        video_uploader.change(
            fn=gradio_video_upload,
            inputs=video_uploader,
            outputs=[upload_status, srt_download, summary_output],
            show_progress=True,
        )

# Launch Gradio app
if __name__ == "__main__":
    try:
        demo.launch(debug=False, show_error=False, pwa=True)
    finally:
        # Clean up temporary files and log final memory usage
        asyncio.run(cleanup_temp_files())
        log_memory_usage("Final")
        tracemalloc.stop()
        logging.info("Gradio app terminated and memory tracking stopped.")