import os
from pyAudioAnalysis import audioBasicIO as aIO
from pyAudioAnalysis import audioSegmentation as aS
import scipy.io.wavfile as wavfile
import numpy as np


"""
A script that iterates through the extracted wav files and uses
pyAudioAnalysis' silence extraction module to make a wav file containing the
segmented audio (when the participant is speaking, silence and virtual
interviewer speech are removed)
"""


def remove_silence(filename, out_dir, smoothing=1.0, weight=0.3, store_segments=False, plot=False):
    """
    A function that implements pyAudioAnalysis' silence extraction module
    and creates wav files of the participant specific portions of audio. The
    smoothing and weight parameters were tuned for the AVEC 2016 dataset.

    Parameters
    ----------
    filename : filepath
        path to the input wav file
    out_dir : filepath
        path to the desired directory (where a participant folder will
        be created containing a 'PXXX_no_silence.wav' file)
    smoothing : float
        tunable parameter to compensate for sparseness of recordings
    weight : float
        probability threshold for silence removal used in SVM
    store_segments : float
        stores no silence segments as wav files within out_dir
    plot : bool
        plots SVM probabilities of silence (used in tuning)

    Returns
    -------
    A folder for each participant containing a single wav file
    (named 'PXXX_no_silence.wav') with the vast majority of silence
    and virtual interviewer speech removed. Feature extraction is
    performed on these segmented wav files.
    """
    participant_id = 'P' + filename.split('/')[-1].split('_')[0]  # PXXX
    if can_segment(participant_id):
        # create participant directory for segmented wav files
        participant_dir = os.path.join(out_dir, participant_id)
        if not os.path.exists(participant_dir):
            os.makedirs(participant_dir)

        [fs, x] = aIO.read_audio_file(filename)
        segments = aS.silence_removal(x, fs, 0.020, 0.020, smooth_window=smoothing, weight=weight, plot=plot)

        out_file = '{}_no_silence.wav'.format(participant_id)
        x_no_silence = np.array([], dtype='int16')
        for s in segments:
            segment_no_silence = x[int(fs * s[0]):int(fs * s[1])]
            if store_segments:  # save no voice segment files within participant directory
                segment_name = "{:s}_{:.2f}-{:.2f}.wav".format(participant_id, s[0], s[1])
                wavfile.write(os.path.join(participant_dir, segment_name), fs, segment_no_silence)
            x_no_silence = np.append(x_no_silence, segment_no_silence)

        # save no silence audio within participant directory
        wavfile.write(os.path.join(participant_dir, out_file), fs, x_no_silence)


def can_segment(participant_id):
    """
    A function that returns True if the participant's interview clip is not
    in the manually identified set of troubled clips. It was not possible to
    segment the clips below were due to excessive static, proximity to the virtual
    interviewer, volume levels, etc.
    """
    troubled = {'P300', 'P305', 'P306', 'P308', 'P315', 'P316', 'P343', 'P354', 'P362', 'P375', 'P378', 'P381', 'P382',
                'P385', 'P387', 'P388', 'P390', 'P392', 'P393', 'P395', 'P408', 'P413', 'P421', 'P438', 'P473', 'P476',
                'P479', 'P490', 'P492'}
    return participant_id not in troubled


if __name__ == '__main__':
    # directory containing raw wav files
    dir_name = '../../data/raw/audio'

    # directory where a participant folder will be created containing their segmented wav file
    out_path = '../../data/interim'

    # iterate through wav files in dir_name and create a segmented wav_file
    for file in os.listdir(dir_name):
        if file.endswith('_P'):  # search in participant folders
            file = os.path.join(file, os.listdir(os.path.join(dir_name, file))[0])
        if file.endswith('.wav'):
            file_name = os.path.join(dir_name, file)
            remove_silence(file_name, out_path)
