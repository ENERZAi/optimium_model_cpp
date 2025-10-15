#include <cmath>
#include <fstream>
#include <cstdio>
#include <string>
#include <string_view>
#include <thread>
#include <vector>
#include <cstring>
#include <atomic>
#include <iostream>
#include "common-whisper.h"
#include "melspecto.h"
#include "token.h"
#include "sampling.h"
#include <Optimium/Runtime.h>
#include <Optimium/Runtime/Logging/LogSettings.h>
#include <Optimium/Runtime/Logging/FileWriter.h>
#include <cstdint>
#include <memory>
#include <filesystem>
#include <chrono>
#define DEBUG_SYMBOL 0

#define GGML_PAD(x, n) (((x) + (n) - 1) & ~((n) - 1))


typedef int32_t whisper_pos;
typedef int32_t whisper_seq_id;

namespace rt = optimium::runtime;

static const std::vector<std::string> non_speech_tokens = {
    "\"", "#", "(", ")", "*", "+", "/", ":", ";", "<", "=", ">", "@", "[", "\\", "]", "^",
    "_", "`", "{", "|", "}", "~", "「", "」", "『", "』", "<<", ">>", "<<<", ">>>", "--",
    "---", "-(", "-[", "('", "(\"", "((", "))", "(((", ")))", "[[", "]]", "{{", "}}", "♪♪",
    "♪♪♪","♩", "♪", "♫", "♬", "♭", "♮", "♯"
};


void save_to_txt(const float* data, int rows, int cols, const std::string& filename) {
    std::ofstream fout(filename);
    if (!fout) {
        std::cerr << "Failed to open file for writing\n";
        return;
    }

    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            fout << data[i * cols + j];
            if (j < cols - 1)
                fout << " ";
        }
        fout << "\n";
    }

    fout.close();
}

bool is_file_exist(const char * filename) {
    std::ifstream infile(filename);
    return infile.good();
}

int whisper_pcm_to_mel_with_state(struct whisper_context * ctx, struct whisper_state * state, const float * samples, int n_samples, int n_threads) {
    int filters_n_mel = 80;
    if (!log_mel_spectrogram(*state, samples, n_samples, WHISPER_SAMPLE_RATE, WHISPER_N_FFT, WHISPER_HOP_LENGTH, ctx->filters.n_mel, n_threads, ctx->filters, false, state->mel)) {
        printf("%s: failed to compute mel spectrogram\n", __func__);
        return -1;
    }

    return 0;
}

constexpr std::string_view help_message = R"(
Usage: whisper_runner [options] <audio_path>

Positional Argument:
  <audio_path>                    Path to the input audio file (e.g., .wav)

Required Options:
  -m, --whisper-model <path>      Path to the main Whisper model file
  -x, --cross_attn_model <path>   Path to the cross-attention model file
  -e, --encoder_model <path>      Path to the encoder model file
  -d, --decoder_model <path>      Path to the decoder model file
  -c, --audio_context <int>       Audio context size

Optional:
  -t, --n_thread <int>            Number of threads to use during inference (default: 4)
  -h, --help                      Show this help message and exit

Example:
  optimium-whisper-demo  \
    -m models/whisper.bin \
    -t 2 \
    -x models/cross_attn \
    -e models/encoder \
    -d models/decoder \
    sample.wav
)";
const size_t required_option = 5;

#define SET_MODEL_PATH(PATH, MODEL)                             \
do {                                                            \
    if (i + 1 < argc && argv[i + 1][0] != '-'){                 \
        PATH = argv[i+1];                                       \
    } else {                                                    \
        std::cerr << "error : missing " << MODEL << " path \n"; \
        return -1;                                              \
    }                                                           \
} while(0)

#define SHOW_HELP_MESSAGE()                     \
do {                                            \
    std::cerr << help_message << std::endl;     \
    return -1;                                  \
} while(0)


int main(int argc, char ** argv){
    if(argc < required_option * 2){
        SHOW_HELP_MESSAGE();
    }

    const char* fname_inp = nullptr;
    const char* path_model = nullptr;
    const char* path_encoder = nullptr;
    const char* path_decoder = nullptr;
    const char* path_cross = nullptr;
    size_t input_audio_ctx = 0;
    int n_thread = 4;

    // parse argv
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];

        if (arg[0] != '-') {
            // Assume this is the input filename
            fname_inp = argv[i];
        } else {
            if (arg == "--n_thread" || arg == "-t") {
                if (i + 1 < argc && argv[i + 1][0] != '-') {
                    n_thread = std::atoi(argv[i + 1]);
                    if (n_thread <= 0) {
                        std::printf("warning: invalid n_thread '%s', using default 4\n", argv[i + 1]);
                        n_thread = 4;
                    }
                } else {
                    std::cerr << "warning: missing value for --n_thread, using default 4\n";
                }
            } else if (arg == "--whisper_model" || arg == "-m") {
                SET_MODEL_PATH(path_model, "whisper");
            } else if (arg == "--cross_attn_model" || arg == "-x") {
                SET_MODEL_PATH(path_cross, "cross attention");
            } else if (arg == "--encoder_model" || arg == "-e") {
                SET_MODEL_PATH(path_encoder, "encoder");
            } else if (arg == "--decoder_model" || arg == "-d") {
                SET_MODEL_PATH(path_decoder, "decoder");
            } else if (arg == "--audio_context" || arg == "-c") {
                if (i + 1 < argc && argv[i + 1][0] != '-') {
                    input_audio_ctx = std::atoi(argv[i + 1]);
                    if (input_audio_ctx <= 0) {
                        std::cerr << "warning: invalid n_thread '%s', using default 4\n", argv[i + 1];
                        return -1;
                    }
                } else {
                    std::cerr << "error: missing value for --audio_context\n";
                    return -1;
                }
            } else {
                SHOW_HELP_MESSAGE();
            }
            ++i; // skip next (it's the value)
        }
    }

    if (fname_inp == nullptr || path_model == nullptr 
        || path_encoder == nullptr 
        || path_decoder == nullptr 
        || path_cross == nullptr 
        || input_audio_ctx == 0) {
        SHOW_HELP_MESSAGE();
    }

    // additional parameters
    float temperature = 0.;
    float temperature_inc = 1.2; //0.2;
    int max_tokens = 0;
    bool no_timestamps = true;
    bool single_segment = false;
    bool completed = false;
    float logprob_thold =  -1.00f;
    float no_speech_thold = 0.6f;
    float word_thold = 0.01f;
    float entropy_thold = 2.40f;
    bool tdrz_enable = false;  // speaker turn enable
    bool print_special = false;
    bool print_realtime = true;
    bool print_timestamps = true;
    bool token_timestamps = false;
    bool suppress_blank = true;
    float max_initial_ts = 1.0f;
    bool diarize = false;

    if (!fname_inp) {
        std::cerr << "error: input audio filename is required" << std::endl;
        return -1;
    }
    if (!path_model){
        return -1;
    }

    if (!is_file_exist(fname_inp)){
        std::printf("error: input file not found '%s'\n", fname_inp);
        return -1;
    }
    if (!is_file_exist(path_model)){
        std::printf("error: model file not found '%s'\n", path_model);
    }

    // Optimium runtime setting
    // rt::logging::setLogLevel(rt::LogLevel::Debug);
    // rt::logging::addLogWriter(std::make_unique<rt::logging::FileWriter>("rt.log"));

    rt::AutoInit Init;

    // rt::config::setPrintThreshold(16);

    // Optimium load
    std::cout << "load start" << std::endl;
    // rt::Model encoder = rt::loadModel("/home/root/whisper_opt_rac/whisper-small-encoder_158-500-1thread");
    rt::ModelLoadOptions EncOptions;
    EncOptions.ThreadCount = n_thread;
    // EncOptions.IntermediateSavePath = "/mnt/sda1/whisper_model_store/tensor/encoder1";
    rt::ModelLoadOptions CrOptions;
    CrOptions.ThreadCount = n_thread;
    // CrOptions.IntermediateSavePath = "/mnt/sda1/whisper_model_store/tensor/cross1";
    rt::ModelLoadOptions DecOptions;
    DecOptions.ThreadCount = n_thread;
    // DecOptions.IntermediateSavePath = "/mnt/sda1/whisper_model_store/tensor/decoder1";
    // CrOptions.IntermediateSavePath = "/home/enerzai/Project/whisper_opt_rac/datarecord";
    
    auto enc_load_start = std::chrono::high_resolution_clock::now();
    rt::Model encoder = rt::loadModel(path_encoder, EncOptions);
    std::cout << "encoder loaded" << std::endl;
    auto enc_load_end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> enc_load_ms = enc_load_end - enc_load_start;
    std::cout << "Encoder load time: " << enc_load_ms.count() << " ms\n";

    auto dec_load_start = std::chrono::high_resolution_clock::now();
    rt::Model decoder = rt::loadModel(path_decoder, DecOptions);
    auto dec_load_end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> dec_load_ms = dec_load_end - dec_load_start;
    std::cout << "decoder loaded" << std::endl;
    std::cout << "Decoder load time: " << dec_load_ms.count() << " ms\n";

    auto cross_load_start = std::chrono::high_resolution_clock::now();
    rt::Model cross   = rt::loadModel(path_cross, CrOptions);
    auto cross_load_end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> cross_load_ms = cross_load_end - cross_load_start;
    std::cout << "cross loaded" << std::endl;
    std::cout << "Cross load time: " << cross_load_ms.count() << " ms\n";

    whisper_context* ctx = new whisper_context;

    {
        auto fin = std::ifstream(path_model, std::ios::binary);
        whisper_model_loader loader = {};

        loader.context = &fin;

        loader.read = [](void * ctx, void * output, size_t read_size) {
            std::ifstream * fin = (std::ifstream*)ctx;
            fin->read((char *)output, read_size);
            return read_size;
        };

        loader.eof = [](void * ctx) {
            std::ifstream * fin = (std::ifstream*)ctx;
            return fin->eof();
        };

        loader.close = [](void * ctx) {
            std::ifstream * fin = (std::ifstream*)ctx;
            fin->close();
        };
        if(!whisper_model_load(&loader, *ctx)){
            printf("error: failed to load gguf file\n");
            return -1;
        }
    }

    // whisper_init_state
    whisper_state * state = new whisper_state;
    ctx->state = state;

    auto sequence = std::make_unique<whisper_sequence>();
    sequence->tokens.reserve(ctx->hparams->n_text_ctx);

    std::vector<float> pcmf32;               // mono-channel F32 PCM
    std::vector<std::vector<float>> pcmf32s; // stereo-channel F32 PCM

    if (!::read_audio_data(fname_inp, pcmf32, pcmf32s, false)) {
        std::printf("error: failed to read audio file '%s'\n", fname_inp);
        return -1;
    }

    float* samples = pcmf32.data();
    int n_samples = pcmf32.size();

    if (whisper_pcm_to_mel_with_state(ctx, ctx->state, samples, n_samples, 1) != 0) {
        printf("%s: failed to compute log mel spectrogram\n", __func__);
        return -2;
    }
    
    // main - whisper_full_parallel  (L5527~)
    const int seek_start = 0;
    const int seek_end = ctx->state->mel.n_len_org;

    // if length of spectrogram is less than 100ms (10 frames), then return
    // basically don't process anything that is less than 100ms
    // ref: https://github.com/ggml-org/whisper.cpp/issues/2065
    const int delta_min = 10;

    if (seek_end < seek_start + delta_min) {
        printf("%s: input is too short - %d ms < 100 ms. consider padding the input audio with silence\n", __func__, (seek_end - seek_start)*10);
        return 0;
    }
    
    std::vector<float> temperatures;
    if (temperature_inc > 0.0f) {
        for (float t = temperature; t < 1.0f + 1e-6f; t += temperature_inc) {
            temperatures.push_back(t);
        }
    } else {
        temperatures.push_back(temperature);
    }

    auto & prompt_past = state->prompt_past;
    prompt_past.clear();

    {
        std::vector<whisper_token> prompt_tokens;

        // initial prompt - there is no provided initial prompt
        prompt_tokens.resize(1024);
        int n_needed = whisper_tokenize(ctx, "", prompt_tokens.data(), prompt_tokens.size());
        if (n_needed < 0) {
            prompt_tokens.resize(-n_needed);
            n_needed = whisper_tokenize(ctx, "", prompt_tokens.data(), prompt_tokens.size());
        }
        prompt_tokens.resize(n_needed);
        auto prompt_n_tokens = prompt_tokens.size();
        
        ctx->state->prompt_past;
        // prepend the prompt tokens to the prompt_past
        if (prompt_tokens.data() && prompt_n_tokens > 0) {
            // parse tokens from the pointer
            for (int i = 0; i < prompt_n_tokens; i++) {
                prompt_past.push_back(prompt_tokens.data()[i]);
            }
            std::rotate(prompt_past.begin(), prompt_past.end() - prompt_n_tokens, prompt_past.end());
        }
    }

    // these tokens determine the task that will be performed
    std::vector<whisper_token> prompt_init = { whisper_token_sot(ctx), };
    // if (whisper_is_multilingual(ctx)) {
    //     const int lang_id = whisper_lang_id(params.language);
    //     state->lang_id = lang_id;
    //     prompt_init.push_back(whisper_token_lang(ctx, lang_id));
    //     if (params.translate) {
    //         prompt_init.push_back(whisper_token_translate(ctx));
    //     } else {
    //         prompt_init.push_back(whisper_token_transcribe(ctx));
    //     }
    // }

    int seek = seek_start;

    std::vector<whisper_token> prompt;
    prompt.reserve(whisper_n_text_ctx(ctx));

    auto & result_all = ctx->state->result_all;

    result_all.clear();

    // beam search prepare - pass for now

    while(true){
        // if only 100ms left, then stop
        if (seek + delta_min >= seek_end) {
            break;
        }

        unsigned int n_text_ctx = GGML_PAD(ctx->hparams->n_text_ctx, 256);
        unsigned int n_text_state = (unsigned int) ctx->hparams->n_text_state;
        unsigned int n_text_head = (unsigned int) ctx->hparams->n_text_head;
        unsigned int n_cache_dim = n_text_state * n_text_ctx;
        unsigned int n_text_state_per_head = n_text_state / n_text_head;
        unsigned int n_audio_ctx = ctx->hparams->n_audio_ctx;
        unsigned int n_audio_state = ctx->hparams->n_audio_state;
        std::cout << "n_text_ctx : " << n_text_ctx << "\nn_text_state : " << n_text_state << "\nn_text_head : " << n_text_head << "\nn_cache_dim : " << n_cache_dim << "\nn_text_state_per_head : " << n_text_state_per_head << "\nn_audio_ctx : "; 
        std::cout << n_audio_ctx << "\nn_audio_state : " << n_audio_state << "\n"; 
        n_audio_ctx = input_audio_ctx;

        // cross cache
        // cross cache
        std::unique_ptr<int8_t[]> cc00, cc02, cc10, cc12, cc20, cc22, cc30, cc32, cc40, cc42, cc50, cc52;
        std::unique_ptr<int8_t[]> cc60, cc62, cc70, cc72, cc80, cc82, cc90, cc92, cc100, cc102, cc110, cc112;
        std::unique_ptr<float[]> cc01, cc03, cc11, cc13, cc21, cc23, cc31, cc33, cc41, cc43, cc51, cc53;
        std::unique_ptr<float[]> cc61, cc63, cc71, cc73, cc81, cc83, cc91, cc93, cc101, cc103, cc111, cc113;
        std::unique_ptr<int[]>   cc04, cc14, cc24, cc34, cc44, cc54, cc64, cc74, cc84, cc94, cc104, cc114;
        rt::TypedTensor<int8_t> tns_cc00, tns_cc02, tns_cc10, tns_cc12, tns_cc20, tns_cc22, tns_cc30, tns_cc32, tns_cc40, tns_cc42, tns_cc50, tns_cc52;
        rt::TypedTensor<int8_t> tns_cc60, tns_cc62, tns_cc70, tns_cc72, tns_cc80, tns_cc82, tns_cc90, tns_cc92, tns_cc100, tns_cc102, tns_cc110, tns_cc112;
        rt::TypedTensor<float> tns_cc01, tns_cc03, tns_cc11, tns_cc13, tns_cc21, tns_cc23, tns_cc31, tns_cc33, tns_cc41, tns_cc43, tns_cc51, tns_cc53;
        rt::TypedTensor<float> tns_cc61, tns_cc63, tns_cc71, tns_cc73, tns_cc81, tns_cc83, tns_cc91, tns_cc93, tns_cc101, tns_cc103, tns_cc111, tns_cc113;
        rt::TypedTensor<int> tns_cc04, tns_cc14, tns_cc24, tns_cc34, tns_cc44, tns_cc54, tns_cc64, tns_cc74, tns_cc84, tns_cc94, tns_cc104, tns_cc114;

        auto indata = std::make_unique<float[]>(80 * 2 * n_audio_ctx);
        for(int row=0; row<80; row++){
            memcpy(indata.get() + row * 2*n_audio_ctx, ctx->state->mel.data.data() + row * state->mel.n_len + seek, 2 * n_audio_ctx * sizeof(float));
        }
#if DEBUG_SYMBOL
        save_to_txt(indata.get(), 80, 2*n_audio_ctx, "/home/enerzai/Project/whisper_opt_rac/indata.txt");
#endif
        // run encoder
        {
            auto enc_req_start = std::chrono::high_resolution_clock::now();
            rt::InferRequest enc_request = encoder.createRequest();   
            // TODO - jy - ctx->state->mel.data : (1, 80, 3902) => [:,:,:2*audio_ctx]를 입력해야 함
            rt::TypedTensor<float> enc_input = rt::tensor<float>({1, 80, (unsigned int)2*n_audio_ctx}, indata.get()); // TODO - jy - shape을 transpose해야 함. 임시 shape
            std::vector<rt::Tensor> enc_inputs;
            enc_inputs.emplace_back(enc_input);
            auto encoutdata = std::make_unique<float[]>(n_audio_ctx*n_audio_state);
            rt::TypedTensor<float> enc_output = rt::tensor<float>({1,(unsigned int)n_audio_ctx, n_audio_state}, encoutdata.get());
            // rt::TypedTensor<float> enc_output = rt::tensor<float>({1,(unsigned int)n_audio_ctx, 768}, encoutdata.get());
            std::vector<rt::Tensor> enc_outputs;
            enc_outputs.emplace_back(enc_output);
            auto enc_req_end = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double, std::milli> enc_req_ms = enc_req_end - enc_req_start;
            std::cout << "Encoder Request time: " << enc_req_ms.count() << " ms\n";

            auto encoder_start = std::chrono::high_resolution_clock::now();
            enc_request.infer(rt::make_array(enc_inputs), rt::make_array(enc_outputs));
            enc_request.wait();
            auto encoder_end = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double, std::milli> duration_ms = encoder_end - encoder_start;
            std::cout << "Execution time: " << duration_ms.count() << " ms\n";
#if DEBUG_SYMBOL
            save_to_txt(encoutdata.get(), n_audio_ctx, 768, "/home/enerzai/Project/whisper_opt_rac/encdata.txt");
#endif
            // std::cout << enc_output.toString() << "\n";
            auto cross_req_start = std::chrono::high_resolution_clock::now();
            rt::InferRequest cr_request = cross.createRequest();
            std::vector<rt::Tensor> cr_inputs;
            cr_inputs.emplace_back(enc_output);
            // cross cache 0
            cc00 = std::make_unique<int8_t[]>(n_text_state * n_audio_ctx);
            cc01 = std::make_unique<float[]>(n_text_head * n_audio_ctx);
            cc02 = std::make_unique<int8_t[]>(n_text_state * n_audio_ctx);
            cc03 = std::make_unique<float[]>(n_text_state);
            cc04 = std::make_unique<int[]>(n_text_state);
            tns_cc00 = rt::tensor<int8_t>({1,n_text_head,n_audio_ctx,n_text_state_per_head}, cc00.get());  
            tns_cc01 = rt::tensor<float>({1,n_text_head,n_audio_ctx,1}, cc01.get());  
            tns_cc02 = rt::tensor<int8_t>({1, n_text_head, n_text_state_per_head/16, n_audio_ctx, 16}, cc02.get());
            tns_cc03 = rt::tensor<float>({1,n_text_head,1,n_text_state_per_head}, cc03.get()); 
            tns_cc04 = rt::tensor<int>({1,n_text_head,n_text_state_per_head}, cc04.get());
            // cross cache 1
            cc10 = std::make_unique<int8_t[]>(n_text_state * n_audio_ctx);
            cc11 = std::make_unique<float[]>(n_text_head * n_audio_ctx);
            cc12 = std::make_unique<int8_t[]>(n_text_state * n_audio_ctx);
            cc13 = std::make_unique<float[]>(n_text_state);
            cc14 = std::make_unique<int[]>(n_text_state);
            tns_cc10 = rt::tensor<int8_t>({1,n_text_head,n_audio_ctx,n_text_state_per_head}, cc10.get());  
            tns_cc11 = rt::tensor<float>({1,n_text_head,n_audio_ctx,1}, cc11.get());  
            tns_cc12 = rt::tensor<int8_t>({1, n_text_head, n_text_state_per_head/16, n_audio_ctx, 16}, cc12.get());
            tns_cc13 = rt::tensor<float>({1,n_text_head,1,n_text_state_per_head}, cc13.get()); 
            tns_cc14 = rt::tensor<int>({1,n_text_head,n_text_state_per_head}, cc14.get());
            // cross cache 2
            cc20 = std::make_unique<int8_t[]>(n_text_state * n_audio_ctx);
            cc21 = std::make_unique<float[]>(n_text_head * n_audio_ctx);  
            cc22 = std::make_unique<int8_t[]>(n_text_state * n_audio_ctx);
            cc23 = std::make_unique<float[]>(n_text_state);
            cc24 = std::make_unique<int[]>(n_text_state);
            tns_cc20 = rt::tensor<int8_t>({1,n_text_head,n_audio_ctx,n_text_state_per_head}, cc20.get());  
            tns_cc21 = rt::tensor<float>({1,n_text_head,n_audio_ctx,1}, cc21.get());  
            tns_cc22 = rt::tensor<int8_t>({1, n_text_head, n_text_state_per_head/16, n_audio_ctx, 16}, cc22.get());
            tns_cc23 = rt::tensor<float>({1,n_text_head,1,n_text_state_per_head}, cc23.get()); 
            tns_cc24 = rt::tensor<int>({1,n_text_head,n_text_state_per_head}, cc24.get());
            // cross cache 3
            cc30 = std::make_unique<int8_t[]>(n_text_state * n_audio_ctx);
            cc31 = std::make_unique<float[]>(n_text_head * n_audio_ctx);  
            cc32 = std::make_unique<int8_t[]>(n_text_state * n_audio_ctx);
            cc33 = std::make_unique<float[]>(n_text_state);
            cc34 = std::make_unique<int[]>(n_text_state);
            tns_cc30 = rt::tensor<int8_t>({1,n_text_head,n_audio_ctx,n_text_state_per_head}, cc30.get());
            tns_cc31 = rt::tensor<float>({1,n_text_head,n_audio_ctx,1}, cc31.get());
            tns_cc32 = rt::tensor<int8_t>({1, n_text_head, n_text_state_per_head/16, n_audio_ctx, 16}, cc32.get());
            tns_cc33 = rt::tensor<float>({1,n_text_head,1,n_text_state_per_head}, cc33.get()); 
            tns_cc34 = rt::tensor<int>({1,n_text_head,n_text_state_per_head}, cc34.get());
            // cross cache 4
            cc40 = std::make_unique<int8_t[]>(n_text_state * n_audio_ctx);
            cc41 = std::make_unique<float[]>(n_text_head * n_audio_ctx);  
            cc42 = std::make_unique<int8_t[]>(n_text_state * n_audio_ctx);
            cc43 = std::make_unique<float[]>(n_text_state);
            cc44 = std::make_unique<int[]>(n_text_state);
            tns_cc40 = rt::tensor<int8_t>({1,n_text_head,n_audio_ctx,n_text_state_per_head}, cc40.get());
            tns_cc41 = rt::tensor<float>({1,n_text_head,n_audio_ctx,1}, cc41.get());
            tns_cc42 = rt::tensor<int8_t>({1, n_text_head, n_text_state_per_head/16, n_audio_ctx, 16}, cc42.get());
            tns_cc43 = rt::tensor<float>({1,n_text_head,1,n_text_state_per_head}, cc43.get());
            tns_cc44 = rt::tensor<int>({1,n_text_head,n_text_state_per_head}, cc44.get());
            // cross cache 5
            cc50 = std::make_unique<int8_t[]>(n_text_state * n_audio_ctx);
            cc51 = std::make_unique<float[]>(n_text_head * n_audio_ctx);  
            cc52 = std::make_unique<int8_t[]>(n_text_state * n_audio_ctx);
            cc53 = std::make_unique<float[]>(n_text_state);
            cc54 = std::make_unique<int[]>(n_text_state);
            tns_cc50 = rt::tensor<int8_t>({1,n_text_head,n_audio_ctx,n_text_state_per_head}, cc50.get());
            tns_cc51 = rt::tensor<float>({1,n_text_head,n_audio_ctx,1}, cc51.get());
            tns_cc52 = rt::tensor<int8_t>({1, n_text_head, n_text_state_per_head/16, n_audio_ctx, 16}, cc52.get());
            tns_cc53 = rt::tensor<float>({1,n_text_head,1,n_text_state_per_head}, cc53.get());
            tns_cc54 = rt::tensor<int>({1,n_text_head,n_text_state_per_head}, cc54.get());
            // cross cache 6
            cc60 = std::make_unique<int8_t[]>(n_text_state * n_audio_ctx);
            cc61 = std::make_unique<float[]>(n_text_head * n_audio_ctx);  
            cc62 = std::make_unique<int8_t[]>(n_text_state * n_audio_ctx);
            cc63 = std::make_unique<float[]>(n_text_state);
            cc64 = std::make_unique<int[]>(n_text_state);
            tns_cc60 = rt::tensor<int8_t>({1,n_text_head,n_audio_ctx,n_text_state_per_head}, cc60.get());
            tns_cc61 = rt::tensor<float>({1,n_text_head,n_audio_ctx,1}, cc61.get());
            tns_cc62 = rt::tensor<int8_t>({1, n_text_head, n_text_state_per_head/16, n_audio_ctx, 16}, cc62.get());
            tns_cc63 = rt::tensor<float>({1,n_text_head,1,n_text_state_per_head}, cc63.get());
            tns_cc64 = rt::tensor<int>({1,n_text_head,n_text_state_per_head}, cc64.get());
            // cross cache 7
            cc70 = std::make_unique<int8_t[]>(n_text_state * n_audio_ctx);
            cc71 = std::make_unique<float[]>(n_text_head * n_audio_ctx);  
            cc72 = std::make_unique<int8_t[]>(n_text_state * n_audio_ctx);
            cc73 = std::make_unique<float[]>(n_text_state);
            cc74 = std::make_unique<int[]>(n_text_state);
            tns_cc70 = rt::tensor<int8_t>({1,n_text_head,n_audio_ctx,n_text_state_per_head}, cc70.get());
            tns_cc71 = rt::tensor<float>({1,n_text_head,n_audio_ctx,1}, cc71.get());
            tns_cc72 = rt::tensor<int8_t>({1, n_text_head, n_text_state_per_head/16, n_audio_ctx, 16}, cc72.get());
            tns_cc73 = rt::tensor<float>({1,n_text_head,1,n_text_state_per_head}, cc73.get()); 
            tns_cc74 = rt::tensor<int>({1,n_text_head,n_text_state_per_head}, cc74.get());
            // cross cache 8
            cc80 = std::make_unique<int8_t[]>(n_text_state * n_audio_ctx);
            cc81 = std::make_unique<float[]>(n_text_head * n_audio_ctx);  
            cc82 = std::make_unique<int8_t[]>(n_text_state * n_audio_ctx);
            cc83 = std::make_unique<float[]>(n_text_state);
            cc84 = std::make_unique<int[]>(n_text_state);
            tns_cc80 = rt::tensor<int8_t>({1,n_text_head,n_audio_ctx,n_text_state_per_head}, cc80.get());
            tns_cc81 = rt::tensor<float>({1,n_text_head,n_audio_ctx,1}, cc81.get());
            tns_cc82 = rt::tensor<int8_t>({1, n_text_head, n_text_state_per_head/16, n_audio_ctx, 16}, cc82.get());
            tns_cc83 = rt::tensor<float>({1,n_text_head,1,n_text_state_per_head}, cc83.get()); 
            tns_cc84 = rt::tensor<int>({1,n_text_head,n_text_state_per_head}, cc84.get());
            // cross cache 9
            cc90 = std::make_unique<int8_t[]>(n_text_state * n_audio_ctx);
            cc91 = std::make_unique<float[]>(n_text_head * n_audio_ctx);  
            cc92 = std::make_unique<int8_t[]>(n_text_state * n_audio_ctx);
            cc93 = std::make_unique<float[]>(n_text_state);
            cc94 = std::make_unique<int[]>(n_text_state);
            tns_cc90 = rt::tensor<int8_t>({1,n_text_head,n_audio_ctx,n_text_state_per_head}, cc90.get());
            tns_cc91 = rt::tensor<float>({1,n_text_head,n_audio_ctx,1}, cc91.get());
            tns_cc92 = rt::tensor<int8_t>({1, n_text_head, n_text_state_per_head/16, n_audio_ctx, 16}, cc92.get());
            tns_cc93 = rt::tensor<float>({1,n_text_head,1,n_text_state_per_head}, cc93.get());
            tns_cc94 = rt::tensor<int>({1,n_text_head,n_text_state_per_head}, cc94.get());
            // cross cache 10
            cc100 = std::make_unique<int8_t[]>(n_text_state * n_audio_ctx);
            cc101 = std::make_unique<float[]>(n_text_head * n_audio_ctx);  
            cc102 = std::make_unique<int8_t[]>(n_text_state * n_audio_ctx);
            cc103 = std::make_unique<float[]>(n_text_state);
            cc104 = std::make_unique<int[]>(n_text_state);
            tns_cc100 = rt::tensor<int8_t>({1,n_text_head,n_audio_ctx,n_text_state_per_head}, cc100.get());
            tns_cc101 = rt::tensor<float>({1,n_text_head,n_audio_ctx,1}, cc101.get());
            tns_cc102 = rt::tensor<int8_t>({1, n_text_head, n_text_state_per_head/16, n_audio_ctx, 16}, cc102.get());
            tns_cc103 = rt::tensor<float>({1,n_text_head,1,n_text_state_per_head}, cc103.get());
            tns_cc104 = rt::tensor<int>({1,n_text_head,n_text_state_per_head}, cc104.get());
            // cross cache 11
            cc110 = std::make_unique<int8_t[]>(n_text_state * n_audio_ctx);
            cc111 = std::make_unique<float[]>(n_text_head * n_audio_ctx);  
            cc112 = std::make_unique<int8_t[]>(n_text_state * n_audio_ctx);
            cc113 = std::make_unique<float[]>(n_text_state);
            cc114 = std::make_unique<int[]>(n_text_state);
            tns_cc110 = rt::tensor<int8_t>({1,n_text_head,n_audio_ctx,n_text_state_per_head}, cc110.get());  
            tns_cc111 = rt::tensor<float>({1,n_text_head,n_audio_ctx,1}, cc111.get());
            tns_cc112 = rt::tensor<int8_t>({1, n_text_head, n_text_state_per_head/16, n_audio_ctx, 16}, cc112.get());
            tns_cc113 = rt::tensor<float>({1,n_text_head,1,n_text_state_per_head}, cc113.get());
            tns_cc114 = rt::tensor<int>({1,n_text_head,n_text_state_per_head}, cc114.get());
            std::vector<rt::Tensor> cr_outputs;
            cr_outputs.insert(cr_outputs.end(), {tns_cc00, tns_cc01, tns_cc02, tns_cc03, tns_cc04});
            cr_outputs.insert(cr_outputs.end(), {tns_cc10, tns_cc11, tns_cc12, tns_cc13, tns_cc14});
            cr_outputs.insert(cr_outputs.end(), {tns_cc20, tns_cc21, tns_cc22, tns_cc23, tns_cc24});
            cr_outputs.insert(cr_outputs.end(), {tns_cc30, tns_cc31, tns_cc32, tns_cc33, tns_cc34});
            cr_outputs.insert(cr_outputs.end(), {tns_cc40, tns_cc41, tns_cc42, tns_cc43, tns_cc44});
            cr_outputs.insert(cr_outputs.end(), {tns_cc50, tns_cc51, tns_cc52, tns_cc53, tns_cc54});
            cr_outputs.insert(cr_outputs.end(), {tns_cc60, tns_cc61, tns_cc62, tns_cc63, tns_cc64});
            cr_outputs.insert(cr_outputs.end(), {tns_cc70, tns_cc71, tns_cc72, tns_cc73, tns_cc74});
            cr_outputs.insert(cr_outputs.end(), {tns_cc80, tns_cc81, tns_cc82, tns_cc83, tns_cc84});
            cr_outputs.insert(cr_outputs.end(), {tns_cc90, tns_cc91, tns_cc92, tns_cc93, tns_cc94});
            cr_outputs.insert(cr_outputs.end(), {tns_cc100, tns_cc101, tns_cc102, tns_cc103, tns_cc104});
            cr_outputs.insert(cr_outputs.end(), {tns_cc110, tns_cc111, tns_cc112, tns_cc113, tns_cc114});
            auto cr_req_end = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double, std::milli> cr_req_ms = cr_req_end - cross_req_start;
            std::cout << "Cross Reqeust time : " << cr_req_ms.count() << " ms\n";

            auto cr_start = std::chrono::high_resolution_clock::now();
            cr_request.infer(rt::make_array(cr_inputs), rt::make_array(cr_outputs));
            cr_request.wait();
            // std::cout << tns_cc00.toString() << "\n";
            // std::cout << tns_cc01.toString() << "\n";
            // std::cout << tns_cc02.toString() << "\n";
            // std::cout << tns_cc03.toString() << "\n";
            // std::cout << tns_cc04.toString() << "\n";
            auto cr_end = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double, std::milli> cr_ms = cr_end - cr_start;

            std::cout << "Cross Execution time : " << cr_ms.count() << " ms\n";
            // save_to_txt(cc00, n_text_head*n_audio_ctx,n_text_state_per_head, "/home/enerzai/Project/whisper_opt_rac/ccdata.txt");
        }
        // if there is a very short audio segment left to process, we remove any past prompt since it tends
        // to confuse the decoder and often make it repeat or hallucinate stuff
        if (seek > seek_start && seek + 500 >= seek_end) {
            prompt_past.clear();
        }

        // step 2 - run decoder and sampling
        /// free all encoder memory except the input/output and allocate memory for decoder
        std::unique_ptr<int64_t[]> token_input(new int64_t[1]);
        std::unique_ptr<int64_t[]> pos_input(new int64_t[1]);
        // we may increase per need but let it static for now
        std::unique_ptr<float[]> logit_out(new float[ctx->vocab.n_vocab]);
        
        auto dec_req_start = std::chrono::high_resolution_clock::now();
        rt::InferRequest dec_request = decoder.createRequest();
        rt::TypedTensor<int64_t> idx_dec_input = rt::tensor<int64_t>({1,1}, token_input.get());
        rt::TypedTensor<int64_t> pos_dec_input = rt::tensor<int64_t>({1}, pos_input.get());
        std::vector<rt::Tensor> dec_inputs;
        // self kv cache - 0
        auto sc00 = std::make_unique<rt::float16[]>(n_cache_dim);
        auto sc01 = std::make_unique<rt::float16[]>(n_cache_dim);
        rt::TypedTensor<rt::float16> dec_sc00 = rt::tensor<rt::float16>({1,n_text_head,n_text_ctx,n_text_state_per_head}, sc00.get());
        rt::TypedTensor<rt::float16> dec_sc01 = rt::tensor<rt::float16>({1,n_text_head,n_text_ctx,n_text_state_per_head}, sc01.get());
        
        // self kv cache - 1
        auto sc10 = std::make_unique<rt::float16[]>(n_cache_dim);
        auto sc11 = std::make_unique<rt::float16[]>(n_cache_dim);
        rt::TypedTensor<rt::float16> dec_sc10 = rt::tensor<rt::float16>({1,n_text_head,n_text_ctx,n_text_state_per_head}, sc10.get());
        rt::TypedTensor<rt::float16> dec_sc11 = rt::tensor<rt::float16>({1,n_text_head,n_text_ctx,n_text_state_per_head}, sc11.get());
        
        // self kv cache - 2
        auto sc20 = std::make_unique<rt::float16[]>(n_cache_dim);
        auto sc21 = std::make_unique<rt::float16[]>(n_cache_dim);
        rt::TypedTensor<rt::float16> dec_sc20 = rt::tensor<rt::float16>({1,n_text_head,n_text_ctx,n_text_state_per_head}, sc20.get()); 
        rt::TypedTensor<rt::float16> dec_sc21 = rt::tensor<rt::float16>({1,n_text_head,n_text_ctx,n_text_state_per_head}, sc21.get());
        
        // self kv cache - 3
        auto sc30 = std::make_unique<rt::float16[]>(n_cache_dim);
        auto sc31 = std::make_unique<rt::float16[]>(n_cache_dim);
        rt::TypedTensor<rt::float16> dec_sc30 = rt::tensor<rt::float16>({1,n_text_head,n_text_ctx,n_text_state_per_head}, sc30.get());
        rt::TypedTensor<rt::float16> dec_sc31 = rt::tensor<rt::float16>({1,n_text_head,n_text_ctx,n_text_state_per_head}, sc31.get());
        
        // self kv cache - 4
        auto sc40 = std::make_unique<rt::float16[]>(n_cache_dim);
        auto sc41 = std::make_unique<rt::float16[]>(n_cache_dim);
        rt::TypedTensor<rt::float16> dec_sc40 = rt::tensor<rt::float16>({1,n_text_head,n_text_ctx,n_text_state_per_head}, sc40.get());
        rt::TypedTensor<rt::float16> dec_sc41 = rt::tensor<rt::float16>({1,n_text_head,n_text_ctx,n_text_state_per_head}, sc41.get());

        // self kv cache - 5
        auto sc50 = std::make_unique<rt::float16[]>(n_cache_dim);
        auto sc51 = std::make_unique<rt::float16[]>(n_cache_dim);
        rt::TypedTensor<rt::float16> dec_sc50 = rt::tensor<rt::float16>({1,n_text_head,n_text_ctx,n_text_state_per_head}, sc50.get());
        rt::TypedTensor<rt::float16> dec_sc51 = rt::tensor<rt::float16>({1,n_text_head,n_text_ctx,n_text_state_per_head}, sc51.get());
        // self kv cache - 6
        auto sc60 = std::make_unique<rt::float16[]>(n_cache_dim);
        auto sc61 = std::make_unique<rt::float16[]>(n_cache_dim);
        rt::TypedTensor<rt::float16> dec_sc60 = rt::tensor<rt::float16>({1,n_text_head,n_text_ctx,n_text_state_per_head}, sc60.get()); 
        rt::TypedTensor<rt::float16> dec_sc61 = rt::tensor<rt::float16>({1,n_text_head,n_text_ctx,n_text_state_per_head}, sc61.get());
        
        // self kv cache - 7
        auto sc70 = std::make_unique<rt::float16[]>(n_cache_dim);
        auto sc71 = std::make_unique<rt::float16[]>(n_cache_dim);
        rt::TypedTensor<rt::float16> dec_sc70 = rt::tensor<rt::float16>({1,n_text_head,n_text_ctx,n_text_state_per_head}, sc70.get());
        rt::TypedTensor<rt::float16> dec_sc71 = rt::tensor<rt::float16>({1,n_text_head,n_text_ctx,n_text_state_per_head}, sc71.get());
        // self kv cache - 8
        auto sc80 = std::make_unique<rt::float16[]>(n_cache_dim);
        auto sc81 = std::make_unique<rt::float16[]>(n_cache_dim);
        rt::TypedTensor<rt::float16> dec_sc80 = rt::tensor<rt::float16>({1,n_text_head,n_text_ctx,n_text_state_per_head}, sc80.get());
        rt::TypedTensor<rt::float16> dec_sc81 = rt::tensor<rt::float16>({1,n_text_head,n_text_ctx,n_text_state_per_head}, sc81.get());
        
        // self kv cache - 9
        auto sc90 = std::make_unique<rt::float16[]>(n_cache_dim);
        auto sc91 = std::make_unique<rt::float16[]>(n_cache_dim);
        rt::TypedTensor<rt::float16> dec_sc90 = rt::tensor<rt::float16>({1,n_text_head,n_text_ctx,n_text_state_per_head}, sc90.get());
        rt::TypedTensor<rt::float16> dec_sc91 = rt::tensor<rt::float16>({1,n_text_head,n_text_ctx,n_text_state_per_head}, sc91.get());
        
        // self kv cache - 10
        auto sc100 = std::make_unique<rt::float16[]>(n_cache_dim);
        auto sc101 = std::make_unique<rt::float16[]>(n_cache_dim);
        rt::TypedTensor<rt::float16> dec_sc100 = rt::tensor<rt::float16>({1,n_text_head,n_text_ctx,n_text_state_per_head}, sc100.get());
        rt::TypedTensor<rt::float16> dec_sc101 = rt::tensor<rt::float16>({1,n_text_head,n_text_ctx,n_text_state_per_head}, sc101.get());
        
        // self kv cache - 11
        auto sc110 = std::make_unique<rt::float16[]>(n_cache_dim);
        auto sc111 = std::make_unique<rt::float16[]>(n_cache_dim);
        rt::TypedTensor<rt::float16> dec_sc110 = rt::tensor<rt::float16>({1,n_text_head,n_text_ctx,n_text_state_per_head}, sc110.get());
        rt::TypedTensor<rt::float16> dec_sc111 = rt::tensor<rt::float16>({1,n_text_head,n_text_ctx,n_text_state_per_head}, sc111.get());
        rt::TypedTensor<float> dec_output = rt::tensor<float>({1,1,(unsigned int) ctx->vocab.n_vocab}, logit_out.get());
        std::vector<rt::Tensor> dec_outputs;
        dec_inputs.emplace_back(idx_dec_input);
        dec_inputs.emplace_back(pos_dec_input);
        dec_inputs.insert(dec_inputs.end(), {dec_sc00, dec_sc01, tns_cc00, tns_cc01, tns_cc02, tns_cc03, tns_cc04});
        dec_inputs.insert(dec_inputs.end(), {dec_sc10, dec_sc11, tns_cc10, tns_cc11, tns_cc12, tns_cc13, tns_cc14});
        dec_inputs.insert(dec_inputs.end(), {dec_sc20, dec_sc21, tns_cc20, tns_cc21, tns_cc22, tns_cc23, tns_cc24});
        dec_inputs.insert(dec_inputs.end(), {dec_sc30, dec_sc31, tns_cc30, tns_cc31, tns_cc32, tns_cc33, tns_cc34});
        dec_inputs.insert(dec_inputs.end(), {dec_sc40, dec_sc41, tns_cc40, tns_cc41, tns_cc42, tns_cc43, tns_cc44});
        dec_inputs.insert(dec_inputs.end(), {dec_sc50, dec_sc51, tns_cc50, tns_cc51, tns_cc52, tns_cc53, tns_cc54});
        dec_inputs.insert(dec_inputs.end(), {dec_sc60, dec_sc61, tns_cc60, tns_cc61, tns_cc62, tns_cc63, tns_cc64});
        dec_inputs.insert(dec_inputs.end(), {dec_sc70, dec_sc71, tns_cc70, tns_cc71, tns_cc72, tns_cc73, tns_cc74});
        dec_inputs.insert(dec_inputs.end(), {dec_sc80, dec_sc81, tns_cc80, tns_cc81, tns_cc82, tns_cc83, tns_cc84});
        dec_inputs.insert(dec_inputs.end(), {dec_sc90, dec_sc91, tns_cc90, tns_cc91, tns_cc92, tns_cc93, tns_cc94});
        dec_inputs.insert(dec_inputs.end(), {dec_sc100, dec_sc101, tns_cc100, tns_cc101, tns_cc102, tns_cc103, tns_cc104});
        dec_inputs.insert(dec_inputs.end(), {dec_sc110, dec_sc111, tns_cc110, tns_cc111, tns_cc112, tns_cc113, tns_cc114});
        dec_outputs.emplace_back(dec_output);
        auto dec_req_end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> dec_req_ms = dec_req_end - dec_req_start;
        std::cout << "Decoder Request time : " << dec_req_ms.count() << " ms\n";


            int seek_delta;

            for(int it=0; it < (int) temperatures.size(); ++it){

                const float t_cur = temperatures[it];

                int n_decoders_cur = 1;
                // GREEDY for now
                if (t_cur > 0.0f){
                    n_decoders_cur = 5; // best_of
                }

                // TAGS: WHISPER_DECODER_INIT
                sequence->tokens.clear();
                sequence->result_len = 0;
                sequence->sum_logprobs = -INFINITY;
                sequence->avg_logprobs = -INFINITY;
                sequence->sum_logprobs_all = 0.0;
                sequence->entropy = 0.0;
                sequence->score = -INFINITY;

                // not support multiple decoders for now
                seek_delta = 100 * WHISPER_CHUNK_SIZE;
                bool has_ts = false;
                bool failed = false;
                auto & result_len = sequence->result_len;

                const int n_logits = ctx->vocab.id_to_token.size();
                std::vector<float> logprobs(n_logits);
                std::vector<float> probs(n_logits);
                std::vector<float> logits(n_logits); // is this essential??

                // init prompt and kv cache for the current iteration
                {
                    prompt.clear();

                    // if we have already generated some text, use it as a prompt to condition the next generation
                    if (!prompt_past.empty() && t_cur < 0.5f && 16384 > 0) {
                        int n_take = std::min(std::min(16384, whisper_n_text_ctx(ctx)/2), int(prompt_past.size()));

                        prompt = { whisper_token_prev(ctx) };
                        prompt.insert(prompt.begin() + 1, prompt_past.end() - n_take, prompt_past.end());
                    }

                    // init new transcription with sot, language (opt) and task tokens
                    prompt.insert(prompt.end(), prompt_init.begin(), prompt_init.end());

                    // print the prompt
                    printf("\n\n");
                    for (int i = 0; i < (int) prompt.size(); i++) {
                        printf("%s: prompt[%d] = %s\n", __func__, i, ctx->vocab.id_to_token.at(prompt[i]).c_str());
                    }
                    printf("\n\n");

                    // TODO - jy - copy the position and token to run decoder
                    std::cout << prompt.data()[0] << "\n";
                    token_input[0] = prompt.data()[0];
                    pos_input[0] = (int64_t) 0;
                    // TODO - run decoder once
                    auto decoder_start = std::chrono::high_resolution_clock::now();
                    dec_request.infer(rt::make_array(dec_inputs), rt::make_array(dec_outputs));
                    dec_request.wait();
                    auto decoder_end = std::chrono::high_resolution_clock::now();
                    // std::cout << dec_output.toString() << "\n";
                    std::chrono::duration<double, std::milli> dec_ms = decoder_end - decoder_start;
                    std::cout << "Decoder Exe time : " << dec_ms.count() << " ms\n";
#if DEBUG_SYMBOL
                    save_to_txt(logit_out, n_logits, 1, "/home/enerzai/Project/whisper_opt_rac/dec.txt");            
#endif
                    // prompt.emplace_back(50363);
                    prompt.emplace_back(50362);
                    std::cout << prompt.data()[1] << "\n";
                    token_input[0] = prompt.data()[1];
                    pos_input[0] = (int64_t) 1;
                    // TODO - run decoder once
                    decoder_start = std::chrono::high_resolution_clock::now();
                    dec_request.infer(rt::make_array(dec_inputs), rt::make_array(dec_outputs));
                    dec_request.wait();
                    decoder_end = std::chrono::high_resolution_clock::now();
                    // std::cout << dec_output.toString() << "\n";
                    dec_ms = decoder_end - decoder_start;
                    std::cout << "Decoder Exe time : " << dec_ms.count() << " ms\n";;

#if DEBUG_SYMBOL
                    save_to_txt(logit_out, n_logits, 1, "/home/enerzai/Project/whisper_opt_rac/decnot.txt");            
#endif

                    // Calculate no_speech probability after first decode.
                    // This has to be done before any logit filtering. Hence we cannot use the probs from the whisper_process_logits.

                    std::vector<float> logit_out_cpy(logit_out.get(), logit_out.get() + n_logits);
                    whisper_compute_logprobs(logit_out_cpy, n_logits, logprobs); // 복사 발생 - 더 좋은 방법?? TODO - jy
                    whisper_compute_probs(logit_out_cpy, n_logits, logprobs, probs);
                    state->no_speech_prob = probs[whisper_token_nosp(ctx)];

                    whisper_process_logits(*ctx, *state, *sequence, logit_out.get(), probs, logits, logprobs, t_cur, 0,
                            suppress_blank, no_timestamps, max_initial_ts, has_ts, tdrz_enable, seek_delta);

                    // we only use one decoder and thus it is not required
                    // {
                    //     const int64_t t_start_sample_us = ggml_time_us();

                    //     state->decoders[0].i_batch = prompt.size() - 1;

                    //     whisper_process_logits(*ctx, *state, state->decoders[0], params, t_cur);

                    //     for (int j = 1; j < n_decoders_cur; ++j) {
                    //         auto & decoder = state->decoders[j];

                    //         whisper_kv_cache_seq_cp(state->kv_self, 0, j, -1, -1);

                    //         memcpy(decoder.probs.data(),    state->decoders[0].probs.data(),    decoder.probs.size()*sizeof(decoder.probs[0]));
                    //         memcpy(decoder.logits.data(),   state->decoders[0].logits.data(),   decoder.logits.size()*sizeof(decoder.logits[0]));
                    //         memcpy(decoder.logprobs.data(), state->decoders[0].logprobs.data(), decoder.logprobs.size()*sizeof(decoder.logprobs[0]));
                    //     }

                    //     state->t_sample_us += ggml_time_us() - t_start_sample_us;
                    // }

                }

                
                
                for (int i = 0, n_max = whisper_n_text_ctx(ctx)/2 - 4; i < n_max; ++i) {

                    // if (params.strategy == whisper_sampling_strategy::WHISPER_SAMPLING_BEAM_SEARCH) {
                    //     for (auto & bc : bc_per_dec) {
                    //         bc.clear();
                    //     }
                    // }

                    // sampling
                    // TODO: avoid memory allocations, optimize, avoid threads?
                    {
                        std::atomic<int> j_cur(0);

                        auto process = [&]() {
                            while (true) {
                                //GREEDY
                                {
                                    sequence->tokens.push_back(whisper_sample_token(*ctx, probs, logprobs, true));
                                    sequence->sum_logprobs_all += sequence->tokens.back().plog;
                                } break;
                            };
                        };

                        const int n_threads = std::min(n_thread, n_decoders_cur);

                        if (n_threads == 1) {
                            process();
                        } else {
                            std::vector<std::thread> threads(n_threads - 1);

                            for (int t = 0; t < n_threads - 1; ++t) {
                                threads[t] = std::thread(process);
                            }

                            process();

                            for (int t = 0; t < n_threads - 1; ++t) {
                                threads[t].join();
                            }
                        }
                    }


                    // update the decoder state
                    // - check if the sequence is completed
                    // - check if the sequence is failed
                    // - update sliding window based on timestamp tokens
                    for (int j = 0; j < n_decoders_cur; ++j) {
                        {
                            const auto & token = sequence->tokens.back();

                            // timestamp token - update sliding window
                            if (token.id > whisper_token_beg(ctx)) {
                                const int seek_delta_new = 2*(token.id - whisper_token_beg(ctx));

                                // do not allow to go back in time
                                if (has_ts && seek_delta > seek_delta_new && result_len < i) {
                                    printf("%s: decoder %d: failed due to seek_delta (%d > %d)\n", __func__, j, seek_delta, seek_delta_new);
                                    failed = true; // TODO: maybe this is not a failure ?
                                    continue;
                                }

                                seek_delta = seek_delta_new;
                                result_len = i + 1;
                                has_ts = true;
                            }

                            // end of segment
                            if (token.id == whisper_token_eot(ctx) ||  // end of text token
                                (max_tokens > 0 && i >= max_tokens) || // max tokens per segment reached
                                (has_ts && seek + seek_delta + delta_min >= seek_end)       // end of audio reached (100ms)
                            ) {
                                if (result_len == 0 && !no_timestamps) {
                                    if (seek + seek_delta + delta_min >= seek_end) {
                                        result_len = i + 1;
                                    } else {
                                        printf("%s: decoder %d failed (result_len = 0)\n", __func__, j);
                                        failed = true;
                                        continue;
                                    }
                                }

                                if (single_segment || no_timestamps) {
                                    result_len = i + 1;
                                    seek_delta = 100*WHISPER_CHUNK_SIZE;
                                }

                                printf("%s: decoder %d completed\n", __func__, j);
                                completed = true;
                                continue;
                            }
                        }

                        // sometimes, the decoding can get stuck in a repetition loop
                        // this is an attempt to mitigate such cases - we flag the decoding as failed and use a fallback strategy
                        if (i == n_max - 1 && (result_len == 0 || seek_delta < 100*WHISPER_CHUNK_SIZE/2)) {
                            printf("%s: decoder %d: failed due to repetition loop\n", __func__, j);
                            failed = true;
                            continue;
                        }
                    }

                    // obtain logits for the next token
                    {
                        const int n_past = prompt.size() + i;

                        if (failed || completed) {
                            break;
                        }
                        std::cout << sequence->tokens.back().id << "\n";
                        token_input[0] = sequence->tokens.back().id;
                        pos_input[0] = n_past;

                        // TODO - jy - run decoder
                        auto decoder_start = std::chrono::high_resolution_clock::now();
                        dec_request.infer(rt::make_array(dec_inputs), rt::make_array(dec_outputs));
                        dec_request.wait();
                        auto decoder_end = std::chrono::high_resolution_clock::now();
                        std::chrono::duration<double, std::milli> dec_ms = decoder_end - decoder_start;
                        // std::cout << i << "/" << whisper_n_text_ctx(ctx) << "-th step Decoder Exe time : " << dec_ms.count() << " ms\n";
                        
                        whisper_process_logits(*ctx, *state, *sequence, logit_out.get(), probs, logits, logprobs, t_cur, 0,
                            suppress_blank, no_timestamps, max_initial_ts, has_ts, tdrz_enable, seek_delta);
                    }
                }

                // rank the resulting sequences and select the best one
                // but we only take one sequence and thus skip it for now

                bool success = true;
                // was the decoding successful for the current temperature?
                // do fallback only if:
                // - we are not at the last temperature
                if (it != (int) temperatures.size() - 1) {
                    if (failed ||
                        (sequence->avg_logprobs < logprob_thold && state->no_speech_prob < no_speech_thold)) {
                        printf("%s: failed due to avg_logprobs %8.5f < %8.5f and no_speech_prob %8.5f < %8.5f\n", __func__, sequence->avg_logprobs, logprob_thold, state->no_speech_prob, no_speech_thold);
                        success = false;
                        state->n_fail_p++;
                    }
                }

                if (success) {
                    break;
                }

                printf("\n%s: failed to decode with temperature = %.2f\n", __func__, t_cur);
            }

            // step 3 - postprocessing
            // output results through a user-provided callback
            {
                const auto result_len = sequence->result_len;

                const auto & tokens_cur = sequence->tokens;

                const bool is_no_speech = (state->no_speech_prob > no_speech_thold &&
                    sequence->avg_logprobs < logprob_thold);

                // WHISPER_LOG_DEBUG("prompt_init.size() = %d, prompt.size() = %d, result_len = %d, seek_delta = %d\n", prompt_init.size(), prompt.size(), result_len, seek_delta);

                // update prompt_past
                prompt_past.clear();
                if (prompt.front() == whisper_token_prev(ctx)) {
                    prompt_past.insert(prompt_past.end(), prompt.begin() + 1, prompt.end() - prompt_init.size());
                }

                for (int i = 0; i < result_len && !is_no_speech; ++i) {
                    prompt_past.push_back(tokens_cur[i].id);
                }

                if (!tokens_cur.empty()  && !is_no_speech) {
                    int  i0 = 0;
                    auto t0 = seek + 2*(tokens_cur.front().tid - whisper_token_beg(ctx));

                    std::string text;
                    bool speaker_turn_next = false;

                    for (int i = 0; i < (int) tokens_cur.size(); i++) {
                        if (print_special || tokens_cur[i].id < whisper_token_eot(ctx)) {
                            text += whisper_token_to_str(ctx, tokens_cur[i].id);
#if DEBUG_SYMBOL
                            std::cout << " TOken : " << ctx->vocab.id_to_token.at(tokens_cur[i].id) << std::endl;
#endif
                        }

                        // [TDRZ] record if speaker turn was predicted after current segment
                        if (tdrz_enable && tokens_cur[i].id == whisper_token_solm(ctx)) {
                            speaker_turn_next = true;
                        }

                        if (tokens_cur[i].id > whisper_token_beg(ctx) && single_segment) {
                            const auto t1 = seek + 2*(tokens_cur[i].tid - whisper_token_beg(ctx));

                            if (!text.empty()) {
                                const auto tt0 = t0;
                                const auto tt1 = t1;

                                if (print_realtime) {
                                    if (print_timestamps) {
                                        printf("[%s --> %s]  %s\n", to_timestamp(tt0).c_str(), to_timestamp(tt1).c_str(), text.c_str());
                                    } else {
                                        printf("%s", text.c_str());
                                        fflush(stdout);
                                    }
                                }

                                //printf("tt0 = %d, tt1 = %d, text = %s, token = %s, token_id = %d, tid = %d\n", tt0, tt1, text.c_str(), ctx->vocab.id_to_token[tokens_cur[i].id].c_str(), tokens_cur[i].id, tokens_cur[i].tid);

                                result_all.push_back({ tt0, tt1, text, state->no_speech_prob, {}, speaker_turn_next });
                                for (int j = i0; j <= i; j++) {
                                    result_all.back().tokens.push_back(tokens_cur[j]);
                                }

                                int n_new = 1;
                                whisper_print_segment_callback(ctx, state, n_new, no_timestamps, diarize);
                                
                            }
                            text = "";
                            while (i < (int) tokens_cur.size() && tokens_cur[i].id > whisper_token_beg(ctx)) {
                                i++;
                            }
                            i--;
                            t0 = t1;
                            i0 = i + 1;
                            speaker_turn_next = false;
                        }
                    }

                    if (!text.empty()) {
                        const auto t1 = seek + seek_delta;

                        const auto tt0 = t0;
                        const auto tt1 = t1;

                        if (print_realtime) {
                            if (print_timestamps) {
                                printf("[%s --> %s]  %s\n", to_timestamp(tt0).c_str(), to_timestamp(tt1).c_str(), text.c_str());
                            } else {
                                printf("%s", text.c_str());
                                fflush(stdout);
                            }
                        }

                        result_all.push_back({ tt0, tt1, text, state->no_speech_prob, {}, speaker_turn_next });
                        for (int j = i0; j < (int) tokens_cur.size(); j++) {
                            result_all.back().tokens.push_back(tokens_cur[j]);
                        }

                        int n_new = 1;

                        printf("Transcription result: ");
                        whisper_print_segment_callback(ctx, state, n_new, no_timestamps, diarize);
                        printf("\n");
                        fflush(stdout);
                    }
                }

                // ref: https://github.com/ggml-org/whisper.cpp/pull/2629
                const bool single_timestamp_ending = tokens_cur.size() > 1 &&
                    tokens_cur[tokens_cur.size() - 2].id < whisper_token_beg(ctx) &&
                    tokens_cur[tokens_cur.size() - 1].id > whisper_token_beg(ctx);
                if (single_timestamp_ending) {
                    printf("single timestamp ending - skip entire chunk\n");
                    seek_delta = std::min(seek_end - seek, WHISPER_CHUNK_SIZE * 100);
                }

                // update audio window
                seek += seek_delta;

                printf("\nseek = %d, seek_delta = %d\n", seek, seek_delta);
            }
          
    }
    
    printf("DONE!\n");

    delete ctx->state;
    delete ctx;

    return 0;
}
