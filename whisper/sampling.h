#ifndef SAMPLE_H
#define SAMPLE_H
#include "loader.h"
#include "common-whisper.h"
#include "token.h"
#include <vector>
#include <cmath>
#include <random>

struct whisper_sequence {
    std::vector<whisper_token_data> tokens;

    // the accumulated transcription in the current iteration (used to truncate the tokens array)
    int result_len;

    double sum_logprobs_all; // the sum of the log probabilities of the tokens
    double sum_logprobs;     // the sum of the log probabilities of the tokens (first result_len tokens)
    double avg_logprobs;     // the average log probability of the tokens
    double entropy;          // the entropy of the tokens
    double score;            // likelihood rank score
};

whisper_token_data whisper_sample_token(
    whisper_context & ctx,
    std::vector<float> & probs,
    std::vector<float> & logprobs,
               bool   best);
void whisper_compute_logprobs(
    const std::vector<float> & logits,
    const int    n_logits,
    std::vector<float> & logprobs);

void whisper_compute_probs(
    const std::vector<float> & logits,
    const int    n_logits,
    const std::vector<float> & logprobs,
    std::vector<float> & probs);

void whisper_process_logits(
    struct whisper_context & ctx,
        struct whisper_state  & state,
        struct whisper_sequence & sequence,
        float* logit_out,
        std::vector<float> & probs,
        std::vector<float> & logits,
        std::vector<float> & logprobs,
        float temperature, int i_batch,
        bool suppress_blank, bool no_timestamps,
        float max_initial_ts, bool has_ts,
        bool tdrz_enable, int seek_delta);

#endif // SAMPLE_H