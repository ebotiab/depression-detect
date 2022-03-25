import numpy as np
from numpy.lib import stride_tricks
import os
from PIL import Image
import scipy.io.wavfile as wav


"""
This script creates spectrogram matrices from wav files that can be passed \
to the CNN. This was heavily adopted from Frank Zalkow's work.
"""


def stft(sig, frame_size, overlap_fac=0.5, window=np.hanning):
    """
    Short-time Fourier transform of audio signal.
    """
    win = window(frame_size)
    hop_size = int(frame_size - np.floor(overlap_fac * frame_size))
    # zeros at beginning (thus center of 1st window should be for sample nr. 0)
    samples = np.append(np.zeros(int(np.floor(frame_size / 2.0))), sig)
    # cols for windowing
    cols = int(np.ceil((len(samples) - frame_size) / float(hop_size)) + 1)
    # zeros at end (thus samples can be fully covered by frames)
    samples = np.append(samples, np.zeros(frame_size))

    frames = stride_tricks.as_strided(samples, shape=(cols, frame_size),
                                      strides=(samples.strides[0] * hop_size,
                                               samples.strides[0])).copy()
    frames *= win

    return np.fft.rfft(frames)


def log_scale_spec(spec, sr=44100, factor=20.):
    """
    Scale frequency axis logarithmically.
    """
    time_bins, freq_bins = np.shape(spec)

    scale = np.linspace(0, 1, freq_bins) ** factor
    scale *= (freq_bins - 1) / max(scale)
    scale = np.unique(np.round(scale))
    scale = scale.astype(int)

    # create spectrogram with new freq bins
    new_spec = np.complex128(np.zeros([time_bins, len(scale)]))
    for i in range(0, len(scale)):
        if i == len(scale)-1:
            new_spec[:, i] = np.sum(spec[:, scale[i]:], axis=1)
        else:
            new_spec[:, i] = np.sum(spec[:, scale[i]:scale[i + 1]], axis=1)

    # list center freq of bins
    allfreqs = np.abs(np.fft.fftfreq(freq_bins * 2, 1. / sr)[:freq_bins + 1])
    freqs = []
    for i in range(0, len(scale)):
        if i == len(scale)-1:
            freqs += [np.mean(allfreqs[scale[i]:])]
        else:
            freqs += [np.mean(allfreqs[scale[i]:scale[i+1]])]

    return new_spec, freqs


def stft_matrix(audio_path, bin_size=2**10, png_name='tmp.png',
                save_png=False, offset=0):
    """
    A function that converts a wav file into a spectrogram represented by a \
    matrix where rows represent frequency bins, columns represent time, and \
    the values of the matrix represent the decibel intensity. A matrix of \
    this form can be passed as input to the CNN after undergoing normalization.
    """
    samplerate, samples = wav.read(audio_path)
    s = stft(samples, bin_size)

    sshow, freq = log_scale_spec(s, sr=samplerate, factor=1)
    ims = 20.*np.log10(np.abs(sshow)/10e-6)  # amplitude to decibel
    time_bins, freq_bins = np.shape(ims)

    ims = np.transpose(ims)
    ims = np.flipud(ims)  # weird - not sure why it needs flipping

    if save_png:
        create_png(ims, png_name)

    return ims


def create_png(im_matrix, png_name):
    """
    Save grayscale png of spectrogram.
    """
    image = Image.fromarray(im_matrix)
    image = image.convert('L')  # convert to grayscale
    image.save(png_name)


if __name__ == '__main__':
    # directory containing participant folders with segmented wav files
    dir_name = '../../data/interim'

    # walks through wav files in dir_name and creates pngs of the spectrogram's.
    # This is a visual representation of what is passed to the CNN before
    # normalization, although a cropped matrix representation is actually
    # passed.
    for subdir, dirs, files in os.walk(dir_name):
        for file in files:
            if file.endswith('.wav'):
                wav_file = os.path.join(subdir, file)
                png_path = subdir + '/' + file[:-4] + '.png'
                print('Processing ' + file + '...')
                stft_matrix(wav_file, png_name=png_path, save_png=True)
