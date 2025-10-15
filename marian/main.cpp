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
#include <Optimium/Runtime.h>
#include <Optimium/Runtime/Logging/LogSettings.h>
#include <Optimium/Runtime/Logging/FileWriter.h>
#include <cstdint>
#include <memory>
#include <filesystem>
#include <chrono>
#include <unistd.h>
#define DEBUG_SYMBOL 0

namespace rt = optimium::runtime;

int arg_max(int n, float* data){
    float max_data = -99999999;
    int max_indice = -1;
    for(int i = 0; i < n; i++){
        if(data[i] > max_data){
            max_indice = i;
            max_data = data[i];
        }
    }
    return max_indice;
}

bool is_file_exist(const char * filename) {
    std::ifstream infile(filename);
    return infile.good();
}

constexpr std::string_view help_message = R"(
Usage: whisper_runner [options] <audio_path>

Required Options:
  -x, --cross_attn_model <path>         Path to the cross-attention model file
  -e, --encoder_model <path>            Path to the encoder model file
  -d, --decoder_model <path>            Path to the decoder model file
  -w, --embedding_weight <path>         Path to the embedding_weight file
  -s, --embedding_weight_scale <path>   Path to the embedding_weight_scale file
  -z, --embedding_weight_zp <path>      Path to the embedding_weight_zp file
  --token <token>                       Token list (split by '-')

Optional:
  -t, --n_thread <int>            Number of threads to use during inference (default: 4)
  -h, --help                      Show this help message and exit

Example:
  optimium-whisper-demo  \
    -t 2 \
    -x models/cross_attn \
    -e models/encoder \
    -d models/decoder \
    -w embedding_weight.bin \
    -s embedding_weight_scale.bin \
    -z embedding_weight_zp.bin \
    --token 609-1359-6834-2-1065-6373-7223-9-0
)";
const size_t required_option = 6;

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

// kyh] main code
int main(int argc, char ** argv){
    if(argc < required_option * 2){
        SHOW_HELP_MESSAGE();
    }

    const char* path_encoder = nullptr;
    const char* path_decoder = nullptr;
    const char* path_cross = nullptr;
    const char* path_weight = nullptr;
    const char* path_weight_scale = nullptr;
    const char* path_weight_zp = nullptr;
    int n_thread = 4;
    std::string token_arg;

    // parse argv
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];

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
        } else if (arg == "--cross_attn_model" || arg == "-x") {
            SET_MODEL_PATH(path_cross, "cross attention");
        } else if (arg == "--encoder_model" || arg == "-e") {
            SET_MODEL_PATH(path_encoder, "encoder");
        } else if (arg == "--decoder_model" || arg == "-d") {
            SET_MODEL_PATH(path_decoder, "decoder");
        } else if (arg == "--embedding_weight" || arg == "-w") {
            SET_MODEL_PATH(path_weight, "weight");
        } else if (arg == "--embedding_weight_scale" || arg == "-s") {
            SET_MODEL_PATH(path_weight_scale, "weight_scale");
        } else if (arg == "--embedding_weight_zp" || arg == "-z") {
            SET_MODEL_PATH(path_weight_zp, "weight_zp");
        } else if (arg == "--token" || arg == "-t") {
            token_arg = argv[i + 1];
        } else {
            SHOW_HELP_MESSAGE();
        }
        ++i; // skip next (it's the value)
    }

    if (path_encoder == nullptr 
        || path_decoder == nullptr 
        || path_cross == nullptr
        || path_weight == nullptr
        || path_weight_scale == nullptr
        || path_weight_zp == nullptr) {
        SHOW_HELP_MESSAGE();
    }

    // Optimium runtime setting
    // rt::logging::setLogLevel(rt::LogLevel::Debug);
    // rt::logging::addLogWriter(std::make_unique<rt::logging::FileWriter>("rt.log"));
    
    std::vector<int64_t> tokens;
    std::stringstream ss(token_arg);
    std::string item;

    while (std::getline(ss, item, '-')) {
        tokens.push_back(std::stoll(item)); // string -> int64_t
    }

    rt::AutoInit Init;
    // rt::config::setPrintThreshold(16);

    // Optimium load
    std::cout << "load start" << std::endl;
    rt::ModelLoadOptions EncOptions;
    EncOptions.ThreadCount = n_thread;

    rt::ModelLoadOptions CrOptions;
    CrOptions.ThreadCount = n_thread;

    rt::ModelLoadOptions DecOptions;
    DecOptions.ThreadCount = n_thread;

    auto enc_load_start = std::chrono::high_resolution_clock::now();
    rt::Model encoder = rt::loadModel(path_encoder, EncOptions);
    auto enc_load_end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> enc_load_ms = enc_load_end - enc_load_start;
    std::cout << "encoder loaded" << std::endl;
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

    // beam search prepare - pass for now
    unsigned int n_text_ctx = 400;
    unsigned int n_text_state = 512;
    unsigned int n_text_head = 8;
    unsigned int n_cache_dim = n_text_state*n_text_ctx;
    unsigned int n_text_state_per_head = n_text_state / n_text_head;
    unsigned int n_audio_state = 512;
    unsigned int token_len = 65008;
    unsigned int seqlen = tokens.size();
    
    rt::TypedTensor<uint8_t> weight = rt::tensor<uint8_t>({(token_len*n_audio_state)/(64*2), 64}, 0);
    weight.load(path_weight);
    rt::TypedTensor<float> weight_scale = rt::tensor<float>({(token_len*n_audio_state)/64, 1}, 0);
    weight_scale.load(path_weight_scale);
    rt::TypedTensor<float> weight_zp = rt::tensor<float>({(token_len*n_audio_state)/64, 1}, 0);
    weight_zp.load(path_weight_zp);

    // demo loop start
    // cross cache
    std::unique_ptr<int8_t[]> cc00, cc02, cc10, cc12, cc20, cc22, cc30, cc32, cc40, cc42, cc50, cc52;
    std::unique_ptr<int8_t[]> cc60, cc62, cc70, cc72;
    std::unique_ptr<float[]> cc01, cc03, cc11, cc13, cc21, cc23, cc31, cc33, cc41, cc43, cc51, cc53;
    std::unique_ptr<float[]> cc61, cc63, cc71, cc73;
    std::unique_ptr<int[]>   cc04, cc14, cc24, cc34, cc44, cc54, cc64, cc74, cc84, cc94, cc104, cc114;
    rt::TypedTensor<int8_t> tns_cc00, tns_cc02, tns_cc10, tns_cc12, tns_cc20, tns_cc22, tns_cc30, tns_cc32, tns_cc40, tns_cc42, tns_cc50, tns_cc52;
    rt::TypedTensor<int8_t> tns_cc60, tns_cc62, tns_cc70, tns_cc72;
    rt::TypedTensor<float> tns_cc01, tns_cc03, tns_cc11, tns_cc13, tns_cc21, tns_cc23, tns_cc31, tns_cc33, tns_cc41, tns_cc43, tns_cc51, tns_cc53;
    rt::TypedTensor<float> tns_cc61, tns_cc63, tns_cc71, tns_cc73;
    rt::TypedTensor<int> tns_cc04, tns_cc14, tns_cc24, tns_cc34, tns_cc44, tns_cc54, tns_cc64, tns_cc74;
    
    // run encoder
    {
        auto enc_req_start = std::chrono::high_resolution_clock::now();
        rt::InferRequest enc_request = encoder.createRequest();   
        std::cout << std::endl;
        rt::TypedTensor<int64_t> enc_input = rt::tensor<int64_t>({1, seqlen}, tokens.data());
        std::vector<int64_t> seq_len_input_value(seqlen);
        for (int64_t i = 0; i < seqlen; i++) {
            seq_len_input_value[i] = i;
        }
        rt::TypedTensor<int64_t> seq_len_input = rt::tensor<int64_t>({seqlen}, seq_len_input_value.data());
        std::vector<rt::Tensor> enc_inputs;
        enc_inputs.emplace_back(enc_input);
        enc_inputs.emplace_back(seq_len_input);
        enc_inputs.insert(enc_inputs.end(), {weight, weight_scale, weight_zp});
        auto encoutdata = std::make_unique<float[]>(seqlen*512);
        rt::TypedTensor<float> enc_output = rt::tensor<float>({1,seqlen, 512}, encoutdata.get());
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
        std::cout << "Encoder Execution time: " << duration_ms.count() << " ms\n";

        auto cross_req_start = std::chrono::high_resolution_clock::now();
        rt::InferRequest cr_request = cross.createRequest();
        std::vector<rt::Tensor> cr_inputs;
        cr_inputs.emplace_back(enc_output);
        // cross cache 0
        cc00 = std::make_unique<int8_t[]>(n_text_state * seqlen);
        cc01 = std::make_unique<float[]>(n_text_head * seqlen);
        cc02 = std::make_unique<int8_t[]>(n_text_state * seqlen);
        cc03 = std::make_unique<float[]>(n_text_state);
        cc04 = std::make_unique<int[]>(n_text_state);
        tns_cc00 = rt::tensor<int8_t>({1,n_text_head,seqlen,n_text_state_per_head}, cc00.get());  
        tns_cc01 = rt::tensor<float>({1,n_text_head,seqlen,1}, cc01.get());  
        tns_cc02 = rt::tensor<int8_t>({1, n_text_head, n_text_state_per_head/16, seqlen, 16}, cc02.get());
        tns_cc03 = rt::tensor<float>({1,n_text_head,1,n_text_state_per_head}, cc03.get()); 
        tns_cc04 = rt::tensor<int>({1,n_text_head,n_text_state_per_head}, cc04.get());
        // cross cache 1
        cc10 = std::make_unique<int8_t[]>(n_text_state * seqlen);
        cc11 = std::make_unique<float[]>(n_text_head * seqlen);
        cc12 = std::make_unique<int8_t[]>(n_text_state * seqlen);
        cc13 = std::make_unique<float[]>(n_text_state);
        cc14 = std::make_unique<int[]>(n_text_state);
        tns_cc10 = rt::tensor<int8_t>({1,n_text_head,seqlen,n_text_state_per_head}, cc10.get());  
        tns_cc11 = rt::tensor<float>({1,n_text_head,seqlen,1}, cc11.get());  
        tns_cc12 = rt::tensor<int8_t>({1, n_text_head, n_text_state_per_head/16, seqlen, 16}, cc12.get());
        tns_cc13 = rt::tensor<float>({1,n_text_head,1,n_text_state_per_head}, cc13.get()); 
        tns_cc14 = rt::tensor<int>({1,n_text_head,n_text_state_per_head}, cc14.get());
        // cross cache 2
        cc20 = std::make_unique<int8_t[]>(n_text_state * seqlen);
        cc21 = std::make_unique<float[]>(n_text_head * seqlen);  
        cc22 = std::make_unique<int8_t[]>(n_text_state * seqlen);
        cc23 = std::make_unique<float[]>(n_text_state);
        cc24 = std::make_unique<int[]>(n_text_state);
        tns_cc20 = rt::tensor<int8_t>({1,n_text_head,seqlen,n_text_state_per_head}, cc20.get());  
        tns_cc21 = rt::tensor<float>({1,n_text_head,seqlen,1}, cc21.get());  
        tns_cc22 = rt::tensor<int8_t>({1, n_text_head, n_text_state_per_head/16, seqlen, 16}, cc22.get());
        tns_cc23 = rt::tensor<float>({1,n_text_head,1,n_text_state_per_head}, cc23.get()); 
        tns_cc24 = rt::tensor<int>({1,n_text_head,n_text_state_per_head}, cc24.get());
        // cross cache 3
        cc30 = std::make_unique<int8_t[]>(n_text_state * seqlen);
        cc31 = std::make_unique<float[]>(n_text_head * seqlen);  
        cc32 = std::make_unique<int8_t[]>(n_text_state * seqlen);
        cc33 = std::make_unique<float[]>(n_text_state);
        cc34 = std::make_unique<int[]>(n_text_state);
        tns_cc30 = rt::tensor<int8_t>({1,n_text_head,seqlen,n_text_state_per_head}, cc30.get());
        tns_cc31 = rt::tensor<float>({1,n_text_head,seqlen,1}, cc31.get());
        tns_cc32 = rt::tensor<int8_t>({1, n_text_head, n_text_state_per_head/16, seqlen, 16}, cc32.get());
        tns_cc33 = rt::tensor<float>({1,n_text_head,1,n_text_state_per_head}, cc33.get()); 
        tns_cc34 = rt::tensor<int>({1,n_text_head,n_text_state_per_head}, cc34.get());
        // cross cache 4
        cc40 = std::make_unique<int8_t[]>(n_text_state * seqlen);
        cc41 = std::make_unique<float[]>(n_text_head * seqlen);  
        cc42 = std::make_unique<int8_t[]>(n_text_state * seqlen);
        cc43 = std::make_unique<float[]>(n_text_state);
        cc44 = std::make_unique<int[]>(n_text_state);
        tns_cc40 = rt::tensor<int8_t>({1,n_text_head,seqlen,n_text_state_per_head}, cc40.get());
        tns_cc41 = rt::tensor<float>({1,n_text_head,seqlen,1}, cc41.get());
        tns_cc42 = rt::tensor<int8_t>({1, n_text_head, n_text_state_per_head/16, seqlen, 16}, cc42.get());
        tns_cc43 = rt::tensor<float>({1,n_text_head,1,n_text_state_per_head}, cc43.get()); 
        tns_cc44 = rt::tensor<int>({1,n_text_head,n_text_state_per_head}, cc44.get());
        // cross cache 5
        cc50 = std::make_unique<int8_t[]>(n_text_state * seqlen);
        cc51 = std::make_unique<float[]>(n_text_head * seqlen);  
        cc52 = std::make_unique<int8_t[]>(n_text_state * seqlen);
        cc53 = std::make_unique<float[]>(n_text_state);
        cc54 = std::make_unique<int[]>(n_text_state);
        tns_cc50 = rt::tensor<int8_t>({1,n_text_head,seqlen,n_text_state_per_head}, cc50.get());
        tns_cc51 = rt::tensor<float>({1,n_text_head,seqlen,1}, cc51.get());
        tns_cc52 = rt::tensor<int8_t>({1, n_text_head, n_text_state_per_head/16, seqlen, 16}, cc52.get());
        tns_cc53 = rt::tensor<float>({1,n_text_head,1,n_text_state_per_head}, cc53.get()); 
        tns_cc54 = rt::tensor<int>({1,n_text_head,n_text_state_per_head}, cc54.get());

        std::vector<rt::Tensor> cr_outputs;
        cr_outputs.insert(cr_outputs.end(), {tns_cc00, tns_cc01, tns_cc02, tns_cc03, tns_cc04});
        cr_outputs.insert(cr_outputs.end(), {tns_cc10, tns_cc11, tns_cc12, tns_cc13, tns_cc14});
        cr_outputs.insert(cr_outputs.end(), {tns_cc20, tns_cc21, tns_cc22, tns_cc23, tns_cc24});
        cr_outputs.insert(cr_outputs.end(), {tns_cc30, tns_cc31, tns_cc32, tns_cc33, tns_cc34});
        cr_outputs.insert(cr_outputs.end(), {tns_cc40, tns_cc41, tns_cc42, tns_cc43, tns_cc44});
        cr_outputs.insert(cr_outputs.end(), {tns_cc50, tns_cc51, tns_cc52, tns_cc53, tns_cc54});
        auto cr_req_end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> cr_req_ms = cr_req_end - cross_req_start;
        std::cout << "Cross Reqeust time : " << cr_req_ms.count() << " ms\n";

        auto cr_start = std::chrono::high_resolution_clock::now();
        cr_request.infer(rt::make_array(cr_inputs), rt::make_array(cr_outputs));
        cr_request.wait();
        auto cr_end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> cr_ms = cr_end - cr_start;

        std::cout << "Cross Execution time : " << cr_ms.count() << " ms\n";
    }

    // step 2 - run decoder and sampling
    std::unique_ptr<int64_t[]> token_input(new int64_t[1]);
    std::unique_ptr<int64_t[]> pos_input(new int64_t[1]);
    std::unique_ptr<float[]> logit_out(new float[65001]);
    
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
    
    rt::TypedTensor<float> dec_output = rt::tensor<float>({1,1,65001}, logit_out.get());
    std::vector<rt::Tensor> dec_outputs;
    dec_inputs.emplace_back(idx_dec_input);
    dec_inputs.emplace_back(pos_dec_input);
    dec_inputs.insert(dec_inputs.end(), {weight, weight_scale, weight_zp});
    dec_inputs.insert(dec_inputs.end(), {dec_sc00, dec_sc01, tns_cc00, tns_cc01, tns_cc02, tns_cc03, tns_cc04});
    dec_inputs.insert(dec_inputs.end(), {dec_sc10, dec_sc11, tns_cc10, tns_cc11, tns_cc12, tns_cc13, tns_cc14});
    dec_inputs.insert(dec_inputs.end(), {dec_sc20, dec_sc21, tns_cc20, tns_cc21, tns_cc22, tns_cc23, tns_cc24});
    dec_inputs.insert(dec_inputs.end(), {dec_sc30, dec_sc31, tns_cc30, tns_cc31, tns_cc32, tns_cc33, tns_cc34});
    dec_inputs.insert(dec_inputs.end(), {dec_sc40, dec_sc41, tns_cc40, tns_cc41, tns_cc42, tns_cc43, tns_cc44});
    dec_inputs.insert(dec_inputs.end(), {dec_sc50, dec_sc51, tns_cc50, tns_cc51, tns_cc52, tns_cc53, tns_cc54});
    dec_outputs.emplace_back(dec_output);
    auto dec_req_end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> dec_req_ms = dec_req_end - dec_req_start;
    std::cout << "Decoder Request time : " << dec_req_ms.count() << " ms\n";
    int64_t max_i = 65000;
    int64_t pos = 0;
    int token_count = 0;
    std::vector<int64_t> result_tensor = {max_i};
    while(token_count < 400){
        token_input[0] = (int64_t) max_i;
        pos_input[0] = pos;
        auto decoder_start = std::chrono::high_resolution_clock::now();
        dec_request.infer(rt::make_array(dec_inputs), rt::make_array(dec_outputs));
        dec_request.wait();
        auto decoder_end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> dec_ms = decoder_end - decoder_start;
        std::cout << "Decoder Exe time : " << dec_ms.count() << " ms\n";
        max_i = (int64_t) arg_max(65001, logit_out.get());
        pos += 1;
        result_tensor.push_back(max_i);
        if(max_i == 0){
            break;
        }
        token_count++;
    }
    std::cout << "output_tensor : ";
    for (size_t i = 0; i < result_tensor.size()-1; i++) {
        std::cout << result_tensor[i] << "-";
    }
    std::cout << result_tensor[result_tensor.size()-1] << "\n";

    printf("\nDONE!\n");
    return 0;
}
