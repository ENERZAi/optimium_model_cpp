#pragma once

#include "loader.h"
#include <string>
#include <vector>
#include <cstdint>

#define WHISPER_SAMPLE_RATE 16000
#define WHISPER_N_FFT       400
#define WHISPER_HOP_LENGTH  160
#define WHISPER_CHUNK_SIZE  30

#define WHISPER_ASSERT(x) \
    do { \
        if (!(x)) { \
            printf("WHISPER_ASSERT: %s:%d: %s\n", __FILE__, __LINE__, #x); \
            abort(); \
        } \
    } while (0)

void whisper_print_segment_callback(struct whisper_context * ctx, struct whisper_state * /*state*/, int n_new, 
    bool no_timestamps, bool diarize);

// Read WAV audio file and store the PCM data into pcmf32
// fname can be a buffer of WAV data instead of a filename
// The sample rate of the audio must be equal to COMMON_SAMPLE_RATE
// If stereo flag is set and the audio has 2 channels, the pcmf32s will contain 2 channel PCM
bool read_audio_data(
        const std::string & fname,
        std::vector<float> & pcmf32,
        std::vector<std::vector<float>> & pcmf32s,
        bool stereo);

// convert timestamp to string, 6000 -> 01:00.000
std::string to_timestamp(int64_t t, bool comma = false);

// given a timestamp get the sample
int timestamp_to_sample(int64_t t, int n_samples, int whisper_sample_rate);

// write text to file, and call system("command voice_id file")
bool speak_with_file(const std::string & command, const std::string & text, const std::string & path, int voice_id);
