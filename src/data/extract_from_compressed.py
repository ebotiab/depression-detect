import fnmatch
import os
import zipfile
import tarfile

"""
A script iterates through a directory of the 189 DAIC-WOZ participant zip
files and extracts the wav and transcript files.
"""


def extract_files(compressed_file, out_dir, is_tar=True, delete_zip=False):
    """
    A function takes in a zip file and extracts the .wav file and
    *TRANSCRIPT.csv files into separate folders in a user
    specified directory.

    Parameters
    ----------
    compressed_file : filepath
        path to the folder containing the DAIC-WOZ zip files
    out_dir : filepath
        path to the desired directory where audio and transcript folders
        will be created
    is_tar : bool
        If true, interprets compressed file as a tar, interprets compressed file as zip otherwise
    delete_zip : bool
        If true, deletes the zip file once relevant files are extracted

    Returns
    -------
    Two directories :
        audio : containing the extracted wav files
        transcripts : containing the extracted transcript csv files
    """
    # create audio directory
    audio_dir = os.path.join(out_dir, 'audio')
    if not os.path.exists(audio_dir):
        os.makedirs(audio_dir)

    # create transcripts directory
    transcripts_dir = os.path.join(out_dir, 'transcripts')
    if not os.path.exists(audio_dir):
        os.makedirs(transcripts_dir)

    if is_tar:  # create object from zip and extract member names
        compressed_ref = tarfile.open(compressed_file, 'r')
        file_names = compressed_ref.getnames()
    else:
        compressed_ref = zipfile.ZipFile(compressed_file)
        file_names = compressed_ref.namelist()

    for f in file_names:  # iterate through files in compressed file
        if f.endswith('.wav'):
            compressed_ref.extract(f, audio_dir)
        elif fnmatch.fnmatch(f.lower(), '*transcript.csv'):
            compressed_ref.extract(f, transcripts_dir)
    compressed_ref.close()

    if delete_zip:
        os.remove(compressed_file)


if __name__ == '__main__':
    # directory containing DAIC-WOZ zip files
    dir_name = '/home/ebbarbera/experiment/'

    # directory where audio and transcripts folders will be created
    out_path = '/../../depression-detect/data/raw'

    # delete zip file after file wav and csv extraction
    remove_zip = False

    # iterate through zip files in dir_name and extracts wav and transcripts
    for file in os.listdir(dir_name):
        if file.endswith('.tar'):
            zip_file = os.path.join(dir_name, file)
            extract_files(zip_file, out_path, is_tar=True, delete_zip=remove_zip)
        elif file.endswith('.zip'):
            zip_file = os.path.join(dir_name, file)
            extract_files(zip_file, out_path, is_tar=False, delete_zip=remove_zip)
