#ifndef TOKEN_H
#define TOKEN_H

#include <vector>
#include <cstring>
#include <string>
#include <regex>
typedef int32_t whisper_token;

typedef struct whisper_token_data {
    whisper_token id;  // token id
    whisper_token tid; // forced timestamp token id

    float p;           // probability of the token
    float plog;        // log probability of the token
    float pt;          // probability of the timestamp token
    float ptsum;       // sum of probabilities of all timestamp tokens

    // token-level timestamp data
    // do not use if you haven't computed token-level timestamps
    int64_t t0;        // start time of the token
    int64_t t1;        //   end time of the token

    // [EXPERIMENTAL] Token-level timestamps with DTW
    // do not use if you haven't computed token-level timestamps with dtw
    // Roughly corresponds to the moment in audio in which the token was output
    int64_t t_dtw;

    float vlen;        // voice length of the token
} whisper_token_data;

struct whipser_context;

const char * whisper_token_to_str(struct whisper_context * ctx, whisper_token token);

whisper_token whisper_token_eot(struct whisper_context * ctx);

whisper_token whisper_token_sot(struct whisper_context * ctx);

whisper_token whisper_token_solm(struct whisper_context * ctx);

whisper_token whisper_token_prev(struct whisper_context * ctx);

whisper_token whisper_token_nosp(struct whisper_context * ctx);

whisper_token whisper_token_not(struct whisper_context * ctx);

whisper_token whisper_token_beg(struct whisper_context * ctx);

whisper_token whisper_token_lang(struct whisper_context * ctx, int lang_id);

int whisper_tokenize(struct whisper_context * ctx, const char * text, whisper_token * tokens, int n_max_tokens);

#endif // TOKEN_H