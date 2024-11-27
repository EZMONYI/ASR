import sys
import os
import json
import argparse
import random
from tqdm import tqdm
import librosa
import numpy as np
import torch
import torchaudio
import re
from transformers import Qwen2AudioForConditionalGeneration, AutoProcessor
from multiprocessing import Process, current_process, set_start_method

def find_audios(parent_dir, exts=['.wav', '.mp3', '.flac', '.webm', '.mp4', '.m4a']):
    """
    Recursively find all audio files within the parent directory with specified extensions.

    Args:
        parent_dir (str): The directory to search for audio files.
        exts (list, optional): List of file extensions to include. Defaults to common audio formats.

    Returns:
        list: A list of full paths to audio files.
    """
    audio_files = []
    for root, dirs, files in os.walk(parent_dir):
        for file in files:
            if os.path.splitext(file)[1].lower() in exts:
                audio_files.append(os.path.join(root, file))
    return audio_files

def save_filelist(audio_files, output_dir):
    """
    Save the list of audio files to filelist.txt in the output directory.

    Args:
        audio_files (list): List of audio file paths.
        output_dir (str): Directory to save the filelist.txt.
    """
    filelist_path = os.path.join(output_dir, "filelist.txt")
    with open(filelist_path, "w") as f:
        for audio in audio_files:
            f.write(f"{audio}\n")
    print(f"File list saved to {filelist_path}")

def load_filelist(output_dir):
    """
    Load the list of audio files from filelist.txt in the output directory.

    Args:
        output_dir (str): Directory where filelist.txt is saved.

    Returns:
        list: List of audio file paths.
    """
    filelist_path = os.path.join(output_dir, "filelist.txt")
    if not os.path.exists(filelist_path):
        print(f"filelist.txt not found in {output_dir}. Cannot resume without file list.")
        sys.exit(1)
    with open(filelist_path, "r") as f:
        audio_files = [line.strip() for line in f if line.strip()]
    print(f"Loaded {len(audio_files)} audio files from {filelist_path}")
    return audio_files

def extract_json(text):
    try:
        json_obj = eval(text)
        return json_obj
    except:
        pass

    # 使用正则表达式查找JSON数据
    json_regex = r'{\s*"(\w+)"\s*:\s*("[^"]*"|\d+|true|false|\{[^{}]*\}|\[[^\[\]]*\])(?:\s*,\s*"\w+"\s*:\s*("[^"]*"|\d+|true|false|\{[^{}]*\}|\[[^\[\]]*\]))*\s*}'
    match = re.search(json_regex, text)

    if match:
        # 如果找到了JSON数据，则解析它并返回Python对象
        json_str = match.group(0)
        json_obj = json.loads(json_str)
        return json_obj
    else:
        # 如果找不到JSON数据，则返回None
        return None

def annotate_audio(gpu_id, audio_subset, output_dir, resume=False):
    """
    Annotate a subset of audio files using a specific GPU.

    Args:
        gpu_id (int): GPU device ID to use.
        audio_subset (list): List of audio file paths to process.
        output_dir (str): Directory to save the JSON annotations.
        resume (bool): Whether to skip already processed files.
    """
    # Set the specific GPU device
    if torch.cuda.is_available():
        device = torch.device(f'cuda:{gpu_id}')
    else:
        device = torch.device('cpu')
    print(f"[{current_process().name}] Using device: {device}")

    # Load processor and model
    processor = AutoProcessor.from_pretrained("Qwen/Qwen2-Audio-7B-Instruct")
    model = Qwen2AudioForConditionalGeneration.from_pretrained("Qwen/Qwen2-Audio-7B-Instruct").to(device)
    print(f"[{current_process().name}] Model loaded successfully.")

    for audio_path in tqdm(audio_subset, desc=f"Processing on GPU {gpu_id}", position=gpu_id):
        try:
            # If resume is enabled, skip already processed files
            if resume:
                audio_filename = os.path.splitext(os.path.basename(audio_path))[0]
                json_filename = f"{audio_filename}.json"
                json_path = os.path.join(output_dir, json_filename)
                if os.path.exists(json_path):
                    continue  # Skip already processed file

            # Prepare conversation
            # single audio template
            conversation = [
                {
                    "role": "user",
                    "content": [
                        {"type": "audio", "audio_url": audio_path},
                        {
                            "type": "text",
                            "text": "What does the person say?"
                        },
                    ]
                },
            ]

            # Load audio
            audio_data, original_sr = librosa.load(audio_path, sr=None, mono=True)

            # Compute total duration in samples
            total_samples = len(audio_data)

            # 30 seconds in samples
            target_samples = original_sr * 30

            # Ensure audio is at least 30 seconds
            if total_samples > target_samples:
                # Randomly select a start point
                start_sample = np.random.randint(0, total_samples - target_samples)
                # Extract 30-second segment
                audio_data = audio_data[start_sample:start_sample + target_samples]
            else:
                print("Audio is shorter than 30 seconds, keeping original length.")
            
            # Resample audio if needed
            target_sr = processor.feature_extractor.sampling_rate
            if original_sr != target_sr:
                audio_data = librosa.resample(audio_data, orig_sr=original_sr, target_sr=target_sr)

            # Apply chat template
            text = processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)

            # Prepare inputs
            inputs = processor(text=text, audios=audio_data, return_tensors="pt", padding=True, sampling_rate=target_sr)
            inputs = {k: v.to(device) for k, v in inputs.items()}

            # Generate response
            generate_ids = model.generate(**inputs, max_length=256)
            generate_ids = generate_ids[:, inputs['input_ids'].size(1):]
            _response = processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]

            # Extract JSON data from response
            response = extract_json(_response)
            if response is None:
                # Save failed JSON to failed directory
                audio_filename = os.path.splitext(os.path.basename(audio_path))[0]
                json_filename = f"{audio_filename}.json"
                json_path = os.path.join(output_dir, 'failed', json_filename)
                with open(json_path, "w") as json_file:
                    json.dump({"path": audio_path, "response": _response}, json_file, indent=4)
                raise ValueError("Failed to extract JSON data from response.")

            # Prepare JSON data
            annotation = {
                "path": audio_path,
                "response": response
            }

            # Save JSON to output directory
            audio_filename = os.path.splitext(os.path.basename(audio_path))[0]
            json_filename = f"{audio_filename}.json"
            json_path = os.path.join(output_dir, json_filename)
            with open(json_path, "w") as json_file:
                json.dump(annotation, json_file, indent=4)

        except Exception as e:
            print(f"[{current_process().name}] Error processing {audio_path}: {e}")

def main():
    parser = argparse.ArgumentParser(description="Audio Annotation Script")
    parser.add_argument("-i", "--input_dir", required=True, help="Input directory containing audio files")
    parser.add_argument("-o", "--output_dir", required=True, help="Output directory to save annotations and filelist.txt")
    parser.add_argument("--resume", action='store_true', help="Resume processing by skipping already annotated files")
    parser.add_argument("--shuffle", action='store_true', help="Shuffle the list of audio files before processing")
    parser.add_argument("--debug", action='store_true', help="Run in single-process debug mode")  # New argument
    args = parser.parse_args()

    input_dir = args.input_dir
    output_dir = args.output_dir
    resume = args.resume
    shuffle = args.shuffle
    debug = args.debug  # Capture the debug flag

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    print(f"Output directory is set to: {output_dir}")

    # add failed directory
    failed_dir = os.path.join(output_dir, "failed")
    os.makedirs(failed_dir, exist_ok=True)
    print(f"Failed directory is set to: {failed_dir}")

    # Determine if resuming or starting fresh
    if resume:
        print("Resume mode enabled. Loading existing file list.")
        audio_files = load_filelist(output_dir)
        print(f"Found {len(audio_files)} audio files in the existing filelist.txt.")

        # remove processed files from the list
        processed_files = [os.path.splitext(os.path.basename(f))[0] for f in os.listdir(output_dir) if f.endswith('.json')]
        processed_files = set(processed_files)
        audio_files = [f for f in audio_files if os.path.splitext(os.path.basename(f))[0] not in processed_files]
        print(f"Remaining {len(audio_files)} audio files to process.")
    else:
        print("Finding all audio files in the input directory.")
        # Find all audio files
        audio_files = find_audios(input_dir)
        print(f"Found {len(audio_files)} audio files.")

        # Save filelist.txt
        save_filelist(audio_files, output_dir)

    # Shuffle audio files if requested
    if shuffle:
        print("Shuffling audio files.")
        random.shuffle(audio_files)

    if debug:
        # Run in single-process debug mode
        print("Running in single-process debug mode.")
        annotate_audio(
            gpu_id=0,  # Default GPU ID for debugging
            audio_subset=audio_files,
            output_dir=output_dir,
            resume=resume
        )
    else:
        # Determine number of GPUs
        num_gpus = torch.cuda.device_count()
        if num_gpus == 0:
            print("No GPUs detected. Using CPU for processing.")
            num_gpus = 1
        else:
            print(f"Number of GPUs available: {num_gpus}")

        # Split audio files among GPUs
        if num_gpus > 1:
            chunks = [[] for _ in range(num_gpus)]
            for idx, audio in enumerate(audio_files):
                chunks[idx % num_gpus].append(audio)
        else:
            chunks = [audio_files]

        # Create and start processes
        processes = []
        for gpu_id in range(num_gpus):
            if gpu_id >= len(chunks) or not chunks[gpu_id]:
                continue  # No more audio files to assign
            p = Process(target=annotate_audio, args=(gpu_id, chunks[gpu_id], output_dir, resume), name=f"Annotator-GPU-{gpu_id}")
            p.start()
            processes.append(p)
            print(f"Started annotator process for GPU {gpu_id}")

        # Wait for all processes to finish
        for p in processes:
            p.join()

    print("All annotations completed.")

if __name__ == "__main__":
    try:
        # Set the multiprocessing start method to 'spawn' to avoid CUDA re-initialization issues
        set_start_method('spawn')
    except RuntimeError:
        # The start method has already been set in another part of the code
        pass
    main()

# python /aifs4su/mmcode/codeclm/Megatron-LMM/codeclm/launcher/scripts/pretrain/exp26/qwen2audio_annotator.py -i /aifs4su/mmdata/rawdata/codeclm/music/txwy100w/NetEaseCloud/ -o /aifs4su/mmdata/rawdata/codeclm/music/txwy100w/metadata/ --shuffle --resume
# python /aifs4su/mmcode/codeclm/Megatron-LMM/codeclm/launcher/scripts/pretrain/exp26/qwen2audio_annotator.py -i /aifs4su/mmdata/rawdata/codeclm/music/cpop49w/audio/ -o /aifs4su/mmdata/rawdata/codeclm/music/cpop49w/metadata/ --shuffle --resume
# python /scratch/buildlam/codeclm/Megatron-LMM/codeclm/launcher/scripts/pretrain/exp26/qwen2audio_annotator.py -i /scratch/buildlam/rawdata/codeclm/music/100_disco10m/audio/ -o /scratch/buildlam/rawdata/codeclm/music/100_disco10m/metadata_tags/ --shuffle --resume
# python /aifs4su/mmcode/codeclm/Megatron-LMM/codeclm/launcher/scripts/pretrain/exp26/qwen2audio_annotator.py -i /aifs4su/mmdata/rawdata/codeclm/music/audio2m/audio/ -o /aifs4su/mmdata/rawdata/codeclm/music/audio2m/metadata/ --shuffle --resumeqwen2audio_annotator.p
