import os
import sys
import azure.cognitiveservices.speech as speechsdk
from pathlib import Path
import time
import argparse

# --- Configuration ---
# Default directories for input and output files.
DEFAULT_INPUT_DIR = Path("input_text")
DEFAULT_OUTPUT_DIR = Path("output")

# --- Global variable to store sentence boundary data from the callback ---
# This list will be populated by the word_boundary_cb function during synthesis.
sentence_boundaries = []

def format_ms_to_mmss(milliseconds_str: str) -> str:
    """
    Helper function to convert a millisecond value (as a string)
    into a formatted MM:SS.ss string.
    Example: "92580.00" -> "01:32.58"
    """
    try:
        # Convert string milliseconds to a float for calculation.
        milliseconds = float(milliseconds_str)
        
        # Calculate total seconds from milliseconds.
        total_seconds = milliseconds / 1000
        
        # Get the whole number of minutes.
        minutes = int(total_seconds // 60)
        
        # Get the remaining seconds (with fractional part).
        seconds = total_seconds % 60
        
        # Format into a "MM:SS.ss" string.
        # MM is zero-padded to 2 digits.
        # SS.ss is zero-padded to 5 characters total (e.g., 07.58).
        return f"{minutes:02d}:{seconds:05.2f}"
    except (ValueError, TypeError):
        # In case of an error, return a default timestamp.
        return "00:00.00"

def word_boundary_cb(evt: speechsdk.SpeechSynthesisWordBoundaryEventArgs):
    """
    Callback function to process word boundary events from the Speech Synthesizer.
    This function is connected to the synthesizer and is called every time a boundary
    (like a word or sentence) is detected in the audio stream.
    """
    global sentence_boundaries
    
    if evt.boundary_type == speechsdk.SpeechSynthesisBoundaryType.Sentence:
        # Convert Azure's tick unit (100ns) to milliseconds.
        start_time_ms = (evt.audio_offset + 5000) / 10000
        duration_ms = evt.duration.total_seconds() * 1000
        end_time_ms = start_time_ms + duration_ms
        
        print(f"Sentence boundary detected: Text='{evt.text}', Start={start_time_ms:.2f}ms, End={end_time_ms:.2f}ms")

        # Store the raw millisecond data. Formatting will be done later.
        sentence_boundaries.append({
            "text": evt.text.strip(),
            "start_ms": f"{start_time_ms:.2f}",
            "end_ms": f"{end_time_ms:.2f}"
        })

def process_script(input_file_path: Path, output_dir: Path = DEFAULT_OUTPUT_DIR):
    """
    Processes a single text file to generate a WAV audio file and a timestamped text file.
    
    Args:
        input_file_path: Path to the input text file
        output_dir: Directory to save output files (default: DEFAULT_OUTPUT_DIR)
    """
    global sentence_boundaries
    # Reset the list for each new file.
    sentence_boundaries = []

    print(f"\n--- Processing file: {input_file_path.name} ---")

    # --- 1. Setup Azure Speech Configuration ---
    try:
        speech_config = speechsdk.SpeechConfig(
            subscription=os.environ.get('SPEECH_KEY'),
            endpoint=os.environ.get('ENDPOINT')
        )
    except TypeError:
        print("Error: 'SPEECH_KEY' or 'ENDPOINT' environment variables not set.")
        return

    speech_config.speech_synthesis_voice_name='zh-HK-HiuMaanNeural'

    speech_config.set_property(
        property_id=speechsdk.PropertyId.SpeechServiceResponse_RequestSentenceBoundary,
        value='true'
    )
    speech_config.speech_synthesis_output_format = speechsdk.SpeechSynthesisOutputFormat.Riff24Khz16BitMonoPcm

    # --- 2. Setup Output Paths ---
    script_output_dir = output_dir / input_file_path.stem
    script_output_dir.mkdir(parents=True, exist_ok=True)
    wav_output_path = str(script_output_dir / f"{input_file_path.stem}.wav")
    print(f"Audio will be saved to: {wav_output_path}")

    # --- 3. Synthesize Speech ---
    audio_config = speechsdk.audio.AudioOutputConfig(filename=wav_output_path)
    speech_synthesizer = speechsdk.SpeechSynthesizer(speech_config=speech_config, audio_config=audio_config)
    speech_synthesizer.synthesis_word_boundary.connect(word_boundary_cb)

    with open(input_file_path, 'r', encoding='utf-8') as f:
        text_to_synthesize = f.read()

    print("Starting speech synthesis...")
    result = speech_synthesizer.speak_text_async(text_to_synthesize).get()
    print("Speech synthesis finished.")

    # --- 4. Check Result and Write Timestamp File ---
    if result.reason == speechsdk.ResultReason.SynthesizingAudioCompleted:
        print("Synthesis completed successfully.")
        
        timestamp_txt_path = script_output_dir / f"{input_file_path.stem}.txt"
        print(f"Saving timestamped sentences to: {timestamp_txt_path}")

        with open(timestamp_txt_path, 'w', encoding='utf-8') as f:
            for item in sentence_boundaries:
                # *** MODIFICATION HERE ***
                # Convert start and end times to the new MM:SS.ss format before writing.
                start_formatted = format_ms_to_mmss(item['start_ms'])
                end_formatted = format_ms_to_mmss(item['end_ms'])
                
                # Format: start_time_end_time_sentence
                line = f"{start_formatted}_{end_formatted}_{item['text']}\n"
                f.write(line)
        print("Timestamp file created successfully.")

    elif result.reason == speechsdk.ResultReason.Canceled:
        cancellation_details = result.cancellation_details
        print(f"Speech synthesis canceled: {cancellation_details.reason}")
        if cancellation_details.reason == speechsdk.CancellationReason.Error:
            print(f"Error details: {cancellation_details.error_details}")

def create_parser():
    """Create command line argument parser"""
    parser = argparse.ArgumentParser(
        description="Text-to-Speech conversion with sentence boundaries",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        '--input', '-i',
        type=str,
        help='Input text file or directory (default: input_text/)'
    )
    
    parser.add_argument(
        '--output', '-o',
        type=str,
        default=str(DEFAULT_OUTPUT_DIR),
        help=f'Output directory (default: {DEFAULT_OUTPUT_DIR})'
    )
    
    return parser

if __name__ == "__main__":
    print("DEBUG: TTS module started")
    print(f"DEBUG: __name__ = {__name__}")
    print(f"DEBUG: sys.argv = {sys.argv}")
    
    parser = create_parser()
    args = parser.parse_args()
    print(f"DEBUG: Parsed arguments - input: {args.input}, output: {args.output}")
    
    # Determine input and output paths
    if args.input:
        input_path = Path(args.input)
    else:
        input_path = DEFAULT_INPUT_DIR
    
    output_dir = Path(args.output)
    output_dir.mkdir(exist_ok=True)
    
    print(f"DEBUG: Input path: {input_path}")
    print(f"DEBUG: Output directory: {output_dir}")
    print(f"DEBUG: Input path exists: {input_path.exists()}")
    print(f"DEBUG: Input path is file: {input_path.is_file() if input_path.exists() else 'N/A'}")
    
    # Check for required environment variables
    speech_key = os.environ.get('SPEECH_KEY')
    endpoint = os.environ.get('ENDPOINT')
    print(f"DEBUG: SPEECH_KEY set: {'Yes' if speech_key else 'No'}")
    print(f"DEBUG: ENDPOINT set: {'Yes' if endpoint else 'No'}")
    
    if not speech_key or not endpoint:
        print("Critical Error: Please set 'SPEECH_KEY' and 'ENDPOINT' environment variables before running.")
        print(f"SPEECH_KEY: {'SET' if speech_key else 'NOT SET'}")
        print(f"ENDPOINT: {'SET' if endpoint else 'NOT SET'}")
        exit(1)

    print("Starting TTS processing...")
    
    # Process input
    if input_path.is_file() and input_path.suffix == '.txt':
        # Process single file
        print(f"Processing single file: {input_path}")
        try:
            process_script(input_path, output_dir)
        except Exception as e:
            print(f"ERROR processing file: {e}")
            import traceback
            traceback.print_exc()
    elif input_path.is_dir():
        # Process all .txt files in directory
        text_files = list(input_path.glob("*.txt"))
        if not text_files:
            print(f"No .txt files found in '{input_path}'.")
        else:
            print(f"Processing {len(text_files)} files from directory: {input_path}")
            for txt_file in text_files:
                try:
                    process_script(txt_file, output_dir)
                    time.sleep(1)
                except Exception as e:
                    print(f"ERROR processing {txt_file}: {e}")
                    import traceback
                    traceback.print_exc()
    else:
        print(f"Error: Input path '{input_path}' is not a valid file or directory.")
        print(f"Current working directory: {Path.cwd()}")
        print(f"Files in current directory: {list(Path.cwd().glob('*'))}")
        exit(1)

    print("\nTTS processing finished.")