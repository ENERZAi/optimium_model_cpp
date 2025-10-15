#include "sampling.h"
#include "loader.h"

void whisper_compute_logprobs(
    const std::vector<float> & logits,
                  const int    n_logits,
          std::vector<float> & logprobs) {
    const float logit_max = *std::max_element(logits.begin(), logits.end());
    float logsumexp = 0.0f;
    for (int i = 0; i < n_logits; ++i) {
        if (logits[i] > -INFINITY) {
            logsumexp += expf(logits[i] - logit_max);
        }
    }
    logsumexp = logf(logsumexp) + logit_max;

    for (int i = 0; i < n_logits; ++i) {
        if (logits[i] > -INFINITY) {
            logprobs[i] = logits[i] - logsumexp;
        } else {
            logprobs[i] = -INFINITY;
        }
    }
}

void whisper_compute_probs(
        const std::vector<float> & logits,
        const int    n_logits,
        const std::vector<float> & logprobs,
        std::vector<float> & probs){
    for (int i = 0; i < n_logits; ++i) {
        if (logits[i] == -INFINITY) {
            probs[i] = 0.0f;
        } else {
            probs[i] = expf(logprobs[i]);
        }
    }
}


whisper_token_data whisper_sample_token(
    whisper_context & ctx,
    std::vector<float> & probs,
    std::vector<float> & logprobs,
               bool   best) {
    whisper_token_data result = {
        0, 0, 0.0f, 0.0f, 0.0f, 0.0f, -1, -1, -1, 0.0f,
    };

    const auto & vocab = ctx.vocab;


    const int n_logits = vocab.n_vocab;

    {
        double sum_ts = 0.0;
        double max_ts = 0.0;

        for (int i = vocab.token_beg; i < n_logits; i++) {
            if (probs[i] == -INFINITY) {
                continue;
            }

            sum_ts += probs[i];
            if (max_ts < probs[i]) {
                max_ts = probs[i];
                result.tid = i;
            }
        }

        result.pt    = max_ts/(sum_ts + 1e-10);
        result.ptsum = sum_ts;
    }

    if (best) {
        for (int i = 0; i < n_logits; ++i) {
            if (result.p < probs[i]) {
                result.id   = i;
                result.p    = probs[i];
                result.plog = logprobs[i];
            }
        }
    } else {
        std::discrete_distribution<> dist(probs.begin(), probs.end());

        std::mt19937 rng(0);
        result.id   = dist(rng);
        result.p    = probs[result.id];
        result.plog = logprobs[result.id];
    }

    if (result.id >= vocab.token_beg) {
        result.tid = result.id;
        result.pt  = result.p;
    }

    return result;
}

// std::vector<whisper_token_data> whisper_sample_token_topk(
//     whisper_context & ctx,
//     std::vector<float>& probs,
//     std::vector<float>& logits,
//     std::vector<float>& logprobs,
//                 int   k) {
//     const auto & vocab = ctx.vocab;


//     const int n_logits = vocab.n_vocab;

//     auto & logits_id = decoder.logits_id;

//     logits_id.resize(n_logits);
//     for (int i = 0; i < n_logits; ++i) {
//     logits_id[i].first = logits[i];
//     logits_id[i].second = i;
//     }

//     {
//     using pair_type = std::remove_reference<decltype(logits_id)>::type::value_type;
//     std::partial_sort(
//             logits_id.begin(),
//             logits_id.begin() + k, logits_id.end(),
//             [](const pair_type & a, const pair_type & b) {
//         return a.first > b.first;
//     });
//     }

//     std::vector<whisper_token_data> result;
//     result.reserve(k);

//     whisper_token tid = vocab.token_beg;

//     float pt    = 0.0;
//     float ptsum = 0.0;

//     {
//         double sum_ts = 0.0;
//         double max_ts = 0.0;

//         for (int i = vocab.token_beg; i < n_logits; i++) {
//             if (probs[i] == -INFINITY) {
//                 continue;
//             }

//             sum_ts += probs[i];
//             if (max_ts < probs[i]) {
//                 max_ts = probs[i];
//                 tid = i;
//             }
//         }

//         pt    = max_ts/(sum_ts + 1e-10);
//         ptsum = sum_ts;
//     }

//     std::discrete_distribution<> dist(probs.begin(), probs.end());

//     for (int i = 0; i < k; ++i) {
//         const auto id = dist(std::mt19937(0));
//         //printf("XXX %d %d %f %f %f %f\n", id, tid, probs[id], logprobs[id], pt, ptsum);

//         result.push_back({ id, tid, probs[id], logprobs[id], pt, ptsum, -1, -1, -1, 0.0f, });

//         if (result[i].id >= vocab.token_beg) {
//             result[i].tid = result[i].id;
//             result[i].pt  = result[i].p;
//         }
//     }

//     return result;
// }




// process the logits for the selected decoder
// - applies logit filters
// - computes logprobs and probs
// TODO: optimize
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
     bool tdrz_enable, int seek_delta) {
    const auto & vocab      = ctx.vocab;
    const auto & tokens_cur = sequence.tokens;

    const bool is_initial = tokens_cur.size() == 0;
    const int  n_logits   = vocab.id_to_token.size();

    WHISPER_ASSERT(n_logits == ctx.vocab.n_vocab);

    // extract the logits for the last token
    // we will be mutating, and therefore we don't want to use the ctx.logits buffer directly
    {
        logits.resize(n_logits);
        memcpy(logits.data(), logit_out + i_batch*n_logits, n_logits*sizeof(float));

        if (temperature > 0.0f) {
            for (int i = 0; i < n_logits; i++) {
                logits[i] /= temperature;
            }
        }

        // will be populated a bit later
        probs.resize(n_logits);
        logprobs.resize(n_logits);
    }

    // apply logit filters here
    // ref: https://github.com/openai/whisper/blob/0b1ba3d46ebf7fe6f953acfd8cad62a4f851b49f/whisper/decoding.py#L480-L493
    {
        // suppress blank
        // https://github.com/openai/whisper/blob/0b1ba3d46ebf7fe6f953acfd8cad62a4f851b49f/whisper/decoding.py#L388-L390
        if (suppress_blank) {
            if (is_initial) {
                logits[vocab.token_eot]           = -INFINITY;
                logits[vocab.token_to_id.at(" ")] = -INFINITY;
            }
        }

        // suppress <|notimestamps|> token
        // ref: https://github.com/openai/whisper/blob/0b1ba3d46ebf7fe6f953acfd8cad62a4f851b49f/whisper/decoding.py#L410-L412
        logits[vocab.token_not] = -INFINITY;
        if (no_timestamps) {
            for (int i = vocab.token_beg; i < n_logits; ++i) {
                logits[i] = -INFINITY;
            }
        }

        // suppress sot and nosp tokens
        logits[vocab.token_sot]  = -INFINITY;
        logits[vocab.token_nosp] = -INFINITY;

        // [TDRZ] when tinydiarize is disabled, suppress solm token
        if (tdrz_enable == false) {
            logits[vocab.token_solm] = -INFINITY;
        }

        // suppress task tokens
        logits[vocab.token_translate]  = -INFINITY;
        logits[vocab.token_transcribe] = -INFINITY;
        logits[vocab.token_prev]       = -INFINITY;

        // suppress lang tokens
        for (size_t i = 0; i < g_lang.size(); ++i) {
            logits[whisper_token_lang(&ctx, i)] = -INFINITY;
        }

        // suppress prev token
        logits[vocab.token_prev] = -INFINITY;

        // if (params.logits_filter_callback) {
        //     params.logits_filter_callback(&ctx, &state, tokens_cur.data(), tokens_cur.size(), logits.data(), params.logits_filter_callback_user_data);
        // }

        // suppress any tokens matching a regular expression
        // ref: https://github.com/openai/whisper/discussions/1041
        // if (params.suppress_regex != nullptr) {
        //     std::regex re(params.suppress_regex);
        //     for (std::pair<whisper_vocab::token, whisper_vocab::id> token_id : vocab.token_to_id) {
        //         if (std::regex_match(token_id.first, re)) {
        //             logits[token_id.second] = -INFINITY;
        //         }
        //     }
        // }

        // suppress non-speech tokens
        // ref: https://github.com/openai/whisper/blob/7858aa9c08d98f75575035ecd6481f462d66ca27/whisper/tokenizer.py#L224-L253
        // if (params.suppress_nst) {
        //     for (const std::string & token : non_speech_tokens) {
        //         const std::string suppress_tokens[] = {token, " " + token};
        //         for (const std::string & suppress_token : suppress_tokens) {
        //             if (vocab.token_to_id.find(suppress_token) != vocab.token_to_id.end()) {
        //                 logits[vocab.token_to_id.at(suppress_token)] = -INFINITY;
        //             }
        //         }
        //     }

        //     // allow hyphens "-" and single quotes "'" between words, but not at the beginning of a word
        //     if (vocab.token_to_id.find(" -") != vocab.token_to_id.end()) {
        //         logits[vocab.token_to_id.at(" -")] = -INFINITY;
        //     }
        //     if (vocab.token_to_id.find(" '") != vocab.token_to_id.end()) {
        //         logits[vocab.token_to_id.at(" '")] = -INFINITY;
        //     }
        // }

        // timestamps have to appear in pairs, except directly before EOT; mask logits accordingly
        // https://github.com/openai/whisper/blob/0b1ba3d46ebf7fe6f953acfd8cad62a4f851b49f/whisper/decoding.py#L414-L424
        {
            const bool last_was_timestamp        = tokens_cur.size() > 0 && tokens_cur.back().id >= vocab.token_beg;
            const bool penultimate_was_timestamp = tokens_cur.size() < 2 || tokens_cur[tokens_cur.size() - 2].id >= vocab.token_beg;

            //WHISPER_LOG_INFO("last_was_timestamp=%d penultimate_was_timestamp=%d\n", last_was_timestamp, penultimate_was_timestamp);

            if (last_was_timestamp) {
                if (penultimate_was_timestamp) {
                    for (int i = vocab.token_beg; i < n_logits; ++i) {
                        logits[i] = -INFINITY;
                    }
                } else {
                    for (int i = 0; i < vocab.token_eot; ++i) {
                        logits[i] = -INFINITY;
                    }
                }
            }
        }

        // the initial timestamp cannot be larger than max_initial_ts
        // ref: https://github.com/openai/whisper/blob/0b1ba3d46ebf7fe6f953acfd8cad62a4f851b49f/whisper/decoding.py#L426-L429
        if (is_initial && max_initial_ts > 0.0f) {
            const float precision = float(WHISPER_CHUNK_SIZE)/ctx.hparams->n_audio_ctx;
            const int   tid0      = std::round(max_initial_ts/precision);

            for (int i = vocab.token_beg + tid0 + 1; i < n_logits; ++i) {
                logits[i] = -INFINITY;
            }
        }

        // condition timestamp tokens to be increasing
        // ref: https://github.com/openai/whisper/pull/831#issuecomment-1385910556
        if (has_ts) {
            const int tid0 = seek_delta/2;

            for (int i = vocab.token_beg; i < vocab.token_beg + tid0; ++i) {
                logits[i] = -INFINITY;
            }
        }

        // populate the logprobs array (log_softmax)
        whisper_compute_logprobs(logits, n_logits, logprobs);

        // if sum of probability over timestamps is above any other token, sample timestamp
        // ref: https://github.com/openai/whisper/blob/0b1ba3d46ebf7fe6f953acfd8cad62a4f851b49f/whisper/decoding.py#L431-L437
        {
            // logsumexp over timestamps
            float timestamp_logprob = -INFINITY;
            {
                float logsumexp = 0.0f;
                const float logprob_max = *std::max_element(logprobs.begin() + vocab.token_beg, logprobs.end());
                for (int i = vocab.token_beg; i < n_logits; ++i) {
                    if (logprobs[i] > -INFINITY) {
                        logsumexp += expf(logprobs[i] - logprob_max);
                    }
                }
                if (logsumexp > 0.0f) {
                    timestamp_logprob = logf(logsumexp) + logprob_max;
                }
            }

            const float max_text_token_logprob = *std::max_element(logprobs.begin(), logprobs.begin() + vocab.token_beg);

            //WHISPER_LOG_INFO("timestamp_logprob=%f max_text_token_logprob=%f\n", timestamp_logprob, max_text_token_logprob);

            if (timestamp_logprob > max_text_token_logprob) {
                for (int i = 0; i < vocab.token_beg; ++i) {
                    logits[i]   = -INFINITY;
                    logprobs[i] = -INFINITY;
                }
            }
        }
    }

    // compute probs
    whisper_compute_probs(logits, n_logits, logprobs, probs);
}