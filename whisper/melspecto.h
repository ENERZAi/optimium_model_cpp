#include <vector>
#include <cstdio>
#include <cassert>
#include <thread>
#include <iostream>
#include <cmath>
#include <algorithm>
#include <fstream>
#include "common-whisper.h"
#include "loader.h"

#define SIN_COS_N_COUNT WHISPER_N_FFT

bool log_mel_spectrogram(
    whisper_state & wstate,
    const float * samples,
    const int   n_samples,
    const int   /*sample_rate*/,
    const int   frame_size,
    const int   frame_step,
    const int   n_mel,
    const int   n_threads,
    const whisper_filters & filters,
    const bool   debug,
    whisper_mel & mel);