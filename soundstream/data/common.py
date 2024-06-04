import torch
from torch.utils.data import Dataset
import soundfile as sf  # Library for reading and writing sound files
from pathlib import Path  # Library for manipulating filesystem paths
import resampy  # Library for resampling audio
import numpy as np  # Library for numerical operations
from typing import Tuple, Union, Optional, List, Dict, Any  # Typing for type hints
import math  # Library for mathematical operations
from tqdm import tqdm  # Library for progress bars
import wave  # Library for reading and writing WAV files

# Define the Base dataset class inheriting from PyTorch's Dataset
class Base(Dataset):
    def __init__(self,
                 data_list: List[Tuple[Path, int]],
                 sampling_rate: int = None,
                 segment_time: int = 3,
                 **kwargs,
                 ):
        super().__init__()
        boundaries = [0]
        self.data_list = []

        # Preprocessing data
        print("Preprocessing data...")
        for filename, sr in tqdm(data_list):
            with wave.open(str(filename), "rb") as audio_file:
                audio_length_frames = audio_file.getnframes()  # Get the number of frames
                sample_rate = audio_file.getframerate()  # Get the sample rate
                audio_length_seconds = audio_length_frames / float(sample_rate)  # Calculate audio length in seconds
            num_chunks = math.ceil(audio_length_seconds / segment_time)  # Calculate the number of chunks
            boundaries.append(boundaries[-1] + num_chunks)  # Append to boundaries list
            self.data_list.append((filename, sr, segment_time))  # Append to data_list

        self.boundaries = np.array(boundaries)  # Convert boundaries to a numpy array
        self.segment_length = segment_time * sampling_rate  # Calculate segment length

    def __len__(self) -> int:
        # Return the total number of segments
        return self.boundaries[-1]

    def _get_file_idx_and_chunk_idx(self, index: int) -> Tuple[int, int]:
        # Get the file index and chunk index from a global index
        bin_pos = np.digitize(index, self.boundaries[1:], right=False)
        chunk_index = index - self.boundaries[bin_pos]
        return bin_pos, chunk_index

    def _get_waveforms(self, index: int, chunk_index: int) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """Get waveform without resampling."""
        wav_file, sr, length_in_time = self.data_list[index]
        offset = int(chunk_index * length_in_time * sr)  # Calculate offset in frames
        frames = int(length_in_time * sr)  # Calculate the number of frames to read

        # Read audio data
        data, _ = sf.read(
            wav_file, start=offset, frames=frames, dtype='float32', always_2d=True)
        data = data.mean(axis=1, keepdims=False)  # Convert to mono by averaging channels
        return data

    def __getitem__(self, index: int) -> torch.Tensor:
        # Get the file and chunk indices
        file_idx, chunk_idx = self._get_file_idx_and_chunk_idx(index)
        data = self._get_waveforms(file_idx, chunk_idx)  # Get the waveform data

        # Resample the data if necessary
        if data.shape[0] != self.segment_length:
            data = resampy.resample(
                data, data.shape[0], self.segment_length, axis=0, filter='kaiser_fast')[:self.segment_length]
            # Uncomment the following lines to pad if the data is shorter than the segment length
            # if data.shape[0] < self.segment_length:
            #     data = np.pad(
            #         data, ((0, self.segment_length - data.shape[0]),), 'constant')
        
        return torch.tensor(data)  # Return the data as a PyTorch tensor
