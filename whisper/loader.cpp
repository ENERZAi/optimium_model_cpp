#define BYTESWAP_VALUE(d) do {} while (0)
#define BYTESWAP_FILTERS(f) do {} while (0)
#define BYTESWAP_TENSOR(t) do {} while (0)
#include "loader.h"
#include "common-whisper.h"

template<typename T>
static void read_safe(whisper_model_loader * loader, T & dest) {
    loader->read(loader->context, &dest, sizeof(T));
    BYTESWAP_VALUE(dest);
}


int whisper_full_n_segments(struct whisper_context * ctx) {
    return ctx->state->result_all.size();
}

int64_t whisper_full_get_segment_t0(struct whisper_context * ctx, int i_segment) {
    return ctx->state->result_all[i_segment].t0;
}

int64_t whisper_full_get_segment_t1(struct whisper_context * ctx, int i_segment) {
    return ctx->state->result_all[i_segment].t1;
}
const char * whisper_full_get_segment_text(struct whisper_context * ctx, int i_segment) {
    return ctx->state->result_all[i_segment].text.c_str();
}

const char * whisper_lang_str(int id) {
    for (const auto & kv : g_lang) {
        if (kv.second.first == id) {
            return kv.first.c_str();
        }
    }

    printf("%s: unknown language id %d\n", __func__, id);
    return nullptr;
}

const char * whisper_lang_str_full(int id) {
   for (const auto & kv : g_lang) {
        if (kv.second.first == id) {
            return kv.second.second.c_str();
        }
    }

    printf("%s: unknown language id %d\n", __func__, id);
 
    return nullptr;
}
bool whisper_model_load(struct whisper_model_loader * loader, whisper_context & wctx) {
    printf("%s: loading model\n", __func__);

    auto & vocab = wctx.vocab;

    // verify magic
    {
        uint32_t magic;
        read_safe(loader, magic);
        if (magic != GGML_FILE_MAGIC) {
            printf("%s: invalid model data (bad magic)\n", __func__);
            return false;
        }
    }

    //load hparams
    whisper_hparams& hparams = *(new whisper_hparams);
    wctx.hparams = &hparams;
    read_safe(loader, hparams.n_vocab);
    hparams.n_vocab -= 1;
    read_safe(loader, hparams.n_audio_ctx);
    read_safe(loader, hparams.n_audio_state);
    read_safe(loader, hparams.n_audio_head);
    read_safe(loader, hparams.n_audio_layer);
    read_safe(loader, hparams.n_text_ctx);
    read_safe(loader, hparams.n_text_state);
    read_safe(loader, hparams.n_text_head);
    read_safe(loader, hparams.n_text_layer);
    read_safe(loader, hparams.n_mels);
    read_safe(loader, hparams.ftype);


    std::string mver = "";

    // for the big tensors, we have the option to store the data in 16-bit floats or quantized
    // in order to save memory and also to speed up the computation

    printf("%s: n_vocab       = %d\n", __func__, hparams.n_vocab);
    printf("%s: n_audio_ctx   = %d\n", __func__, hparams.n_audio_ctx);
    printf("%s: n_audio_state = %d\n", __func__, hparams.n_audio_state);
    printf("%s: n_audio_head  = %d\n", __func__, hparams.n_audio_head);
    printf("%s: n_audio_layer = %d\n", __func__, hparams.n_audio_layer);
    printf("%s: n_text_ctx    = %d\n", __func__, hparams.n_text_ctx);
    printf("%s: n_text_state  = %d\n", __func__, hparams.n_text_state);
    printf("%s: n_text_head   = %d\n", __func__, hparams.n_text_head);
    printf("%s: n_text_layer  = %d\n", __func__, hparams.n_text_layer);
    printf("%s: n_mels        = %d\n", __func__, hparams.n_mels);
    printf("%s: ftype         = %d\n", __func__, hparams.ftype);
    

    // load mel filters
    {
        auto & filters = wctx.filters;

        read_safe(loader, filters.n_mel);
        read_safe(loader, filters.n_fft);

        filters.data.resize(filters.n_mel * filters.n_fft);
        loader->read(loader->context, filters.data.data(), filters.data.size() * sizeof(float));
        BYTESWAP_FILTERS(filters);
    }

    // load vocab
    {
        int32_t n_vocab = 0;
        read_safe(loader, n_vocab);

        //if (n_vocab != model.hparams.n_vocab) {
        //    WHISPER_LOG_ERROR("%s: invalid model file '%s' (bad vocab size %d != %d)\n",
        //            __func__, fname.c_str(), n_vocab, model.hparams.n_vocab);
        //    return false;
        //}

        std::string word;
        std::vector<char> tmp;

        tmp.reserve(128);

        // std::ofstream fout("token.txt");
        // if (!fout.is_open()){
        //     std::cerr << "Failed to open " << std::endl;
        //     return false;
        // }

        for (int i = 0; i < n_vocab; i++) {
            uint32_t len;
            read_safe(loader, len);

            if (len > 0) {
                tmp.resize(len);
                loader->read(loader->context, &tmp[0], tmp.size()); // read to buffer
                word.assign(&tmp[0], tmp.size());
            } else {
                // seems like we have an empty-string token in multi-language models (i = 50256)
                //WHISPER_LOG_WARN("%s: warning: empty-string token in vocab, i = %d\n", __func__, i);
                word = "";
            }
            // fout << "read : " << word << std::endl;
            vocab.token_to_id[word] = i;
            vocab.id_to_token[i] = word;

            //printf("%s: vocab[%d] = '%s'\n", __func__, i, word.c_str());
        }

        // monkey patching - jy !!!!!
        hparams.n_vocab++;
        vocab.n_vocab = hparams.n_vocab;
        if (vocab.is_multilingual()) {
            vocab.token_eot++;
            vocab.token_sot++;

            // account for variable number of language tokens
            const int dt = vocab.num_languages() - 98;

            vocab.token_translate  += dt;
            vocab.token_transcribe += dt;
            vocab.token_solm       += dt;
            vocab.token_prev       += dt;
            vocab.token_nosp       += dt;
            vocab.token_not        += dt;
            vocab.token_beg        += dt;
        }

        if (n_vocab < hparams.n_vocab) {
            printf("%s: adding %d extra tokens\n", __func__, hparams.n_vocab - n_vocab);
            for (int i = n_vocab; i < hparams.n_vocab; i++) {
                if (i > vocab.token_beg) {
                    word = "[_TT_" + std::to_string(i - vocab.token_beg) + "]";
                } else if (i == vocab.token_eot) {
                    word = "[_EOT_]";
                } else if (i == vocab.token_sot) {
                    word = "[_SOT_]";
                } else if (i == vocab.token_translate) {
                    word = "[_TRANSLATE_]";
                } else if (i == vocab.token_transcribe) {
                    word = "[_TRANSCRIBE_]";
                } else if (i == vocab.token_solm) {
                    word = "[_SOLM_]";
                } else if (i == vocab.token_prev) {
                    word = "[_PREV_]";
                } else if (i == vocab.token_nosp) {
                    word = "[_NOSP_]";
                } else if (i == vocab.token_not) {
                    word = "[_NOT_]";
                } else if (i == vocab.token_beg) {
                    word = "[_BEG_]";
                } else if (i > vocab.token_sot && i <= vocab.token_sot + vocab.num_languages()) {
                    word = "[_LANG_" + std::string(whisper_lang_str(i - vocab.token_sot - 1)) + "]";
                } else {
                    word = "[_extra_token_" + std::to_string(i) + "]";
                }
                vocab.token_to_id[word] = i;
                vocab.id_to_token[i] = word;
            }
        }

        printf("%s: n_langs       = %d\n", __func__, vocab.num_languages());
    }

    // const ggml_type wtype = wctx.wtype;
    // const ggml_type vtype = wctx.wtype == GGML_TYPE_F32 ? GGML_TYPE_F32 : GGML_TYPE_F16; // conv type

    // const auto & hparams = model.hparams;

    // const int n_audio_layer = hparams.n_audio_layer;
    // const int n_text_layer  = hparams.n_text_layer;

    // const size_t n_tensors = 10 /* input */ + 15 + 15*n_audio_layer + 24*n_text_layer;

    // std::map<ggml_backend_buffer_type_t, ggml_context *> ctx_map;
    // auto get_ctx = [&](ggml_backend_buffer_type_t buft) -> ggml_context * {
    //     auto it = ctx_map.find(buft);
    //     if (it == ctx_map.end()) {
    //         ggml_init_params params = {
    //             /*.mem_size   =*/ n_tensors * ggml_tensor_overhead(),
    //             /*.mem_buffer =*/ nullptr,
    //             /*.no_alloc   =*/ true,
    //         };

    //         ggml_context * ctx = ggml_init(params);
    //         if (!ctx) {
    //             throw std::runtime_error("failed to create ggml context");
    //         }

    //         ctx_map[buft] = ctx;
    //         model.ctxs.emplace_back(ctx);

    //         return ctx;
    //     }

    //     return it->second;
    // };

    
    
    return true;
}


int whisper_n_len_from_state(struct whisper_state * state) {
    return state->mel.n_len_org;
}

int whisper_n_len(struct whisper_context * ctx) {
    return ctx->state->mel.n_len_org;
}

int whisper_n_vocab(struct whisper_context * ctx) {
    return ctx->vocab.n_vocab;
}

int whisper_n_text_ctx(struct whisper_context * ctx) {
    return ctx->hparams->n_text_ctx;
}

int whisper_n_audio_ctx(struct whisper_context * ctx) {
    return ctx->hparams->n_audio_ctx;
}

int whisper_is_multilingual(struct whisper_context * ctx) {
    return ctx->vocab.is_multilingual() ? 1 : 0;
}