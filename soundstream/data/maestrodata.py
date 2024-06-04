import soundfile as sf  # Library for reading and writing sound files
from pathlib import Path  # Library for manipulating filesystem paths
import json  # Library for working with JSON data
from tqdm import tqdm  # Library for progress bars

# Import the Base class from the common module in the same package
from .common import Base

# Define the Maestro dataset class inheriting from the Base class
class Maestro(Base):
    def __init__(
        self,
        path: str,
        split: str = "train",
        **kwargs
    ):
        path = Path(path)  # Convert the path to a Path object
        meta_file = path / "maestro-v3.0.0.json"  # Define the path to the metadata file
        if split == "val" or split == "valid":
            split = "validation"  # Normalize the validation split name
        with open(meta_file, "r") as f:
            meta = json.load(f)  # Load metadata from the JSON file
        track_ids = [k for k, v in meta["split"].items() if v == split]  # Get track IDs for the specified split
        mapping = {
            t: (path / meta["audio_filename"][t], path / meta["midi_filename"][t]) for t in track_ids
        }  # Create a mapping of track IDs to their audio and MIDI file paths

        data_list = []  # Initialize an empty list to hold data information
        print("Loading Maestro...")
        for track_id, (wav_file, midi_file) in tqdm(mapping.items()):
            if '2018' in str(wav_file):  # Only include files from 2018
                info = sf.info(wav_file)  # Get audio file information
                sr = info.samplerate  # Get the sample rate
                data_list.append((wav_file, sr))  # Append the file and sample rate to data_list

        super().__init__(data_list, **kwargs)  # Initialize the Base class with data_list and other arguments