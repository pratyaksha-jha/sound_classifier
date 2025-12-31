import os
import numpy as np
import librosa
import math
import librosa.display
import json

DATASET_PATH = "genres_original1"
JSON_PATH = "data2.json"
SAMPLE_RATE = 22050
DURATION = 30
SAMPLES_PER_TRACK = SAMPLE_RATE*DURATION


def save_mfcc(dataset_path, json_path, n_mfccs = 13, n_fft = 2048, hop_length = 512, num_segments = 5):
    #num segments increases the number of training data
    data = {
        "mapping" : [],
        "mfcc" : [],
        "labels" : []
    }

    num_samples_per_segment = int(SAMPLES_PER_TRACK/num_segments)
    expected_num_mfcc_vectors_per_segment = math.ceil(num_samples_per_segment/hop_length)

    for i, (dirpath, dirnames, filenames) in enumerate(os.walk(dataset_path)):
        #ensure we are not all the root level
        if dirpath is not dataset_path:
        # Normalize the path to ensure consistent slashes
            normalized_path = os.path.normpath(dirpath)
            # Split using the OS-specific separator (\ for Windows)
            dirpath_components = normalized_path.split(os.sep)     #genre/blues =>["genre", "blues"]
            semantic_label = dirpath_components[-1]
            data["mapping"].append(semantic_label)
            genre_label = len(data["mapping"])-1
            print("\nProcessing {}".format(semantic_label))

            #process files for a specific genre
            for f in filenames:
                file_path = os.path.join(dirpath, f)
                signal, sr = librosa.load(file_path, sr = SAMPLE_RATE)
            
            #extracting mfccs and storing data
                for s in range(num_segments):
                    start_sample = num_samples_per_segment * s
                    finish_sample = num_samples_per_segment+start_sample
                    #store mfcc for segment if it has expected length
                    
                    mfcc = librosa.feature.mfcc(y = signal[start_sample:finish_sample], sr = sr,
                                                n_fft = n_fft, 
                                                n_mfcc=n_mfccs, 
                                                hop_length= hop_length)
                    mfcc = mfcc.T

                    if len(mfcc) > expected_num_mfcc_vectors_per_segment:
                                mfcc = mfcc[:expected_num_mfcc_vectors_per_segment]
                            
                    elif len(mfcc) < expected_num_mfcc_vectors_per_segment:
                                padding = expected_num_mfcc_vectors_per_segment - len(mfcc)
                                mfcc = np.pad(mfcc, ((0, padding), (0, 0)), mode='constant')

                    if len(mfcc) == expected_num_mfcc_vectors_per_segment:
                                data["mfcc"].append(mfcc.tolist())
                                data["labels"].append(genre_label)
                        
                    print(f"File {f} processed.")

    with open(json_path, "w") as fp:
        json.dump(data, fp, indent=4)


if __name__ == "__main__":
   save_mfcc(DATASET_PATH, JSON_PATH, num_segments=10)