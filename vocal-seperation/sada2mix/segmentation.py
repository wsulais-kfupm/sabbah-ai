## Imports
import os
import soundfile as sf
import pandas as pd

## Constants
OUTPUT_PATH = "segmented"
ROOT = "/ibex/user/shiekhmf/sada/original"


def path(path_0,path_1):
    return os.path.join(path_0,path_1)

def create_segments(filename, rows):
    '''
        Create segments of a specific file given filename and its corresponding rows.
        Segments are saved in a folder called audios_test in ROOT.
    '''
    wav_file, samplerate = sf.read(path(ROOT, filename))
    for idx, row in rows.iterrows():
        start_index = int(row['SegmentStart'] * samplerate)
        end_index = int(row['SegmentEnd'] * samplerate)
        segment = wav_file[start_index:end_index]
        sf.write(path(OUTPUT_PATH,f"{row['SegmentID']}.wav"), segment, samplerate)


test_csv_df = pd.read_csv(path(ROOT, 'train.csv')).sort_values('FileName')

# Get unique filenames
unique_file_names = test_csv_df['FileName'].unique()


for idx, filename in enumerate(unique_file_names):
    rows = test_csv_df[test_csv_df['FileName'] == filename]
    rows = rows[rows["SegmentLength"] > 5]
    create_segments(filename,rows)
    
    if idx%10 == 0 and idx != 0:
        print(f"finished {idx} of {len(unique_file_names)} files")
