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

// tokenizer.json 로드 (model->vocab)
bool load_tokenizer_json(const std::string &filename, std::map<std::string, int64_t>& vocab, std::map<int64_t, std::string>& inv_vocab) {
    std::ifstream ifs(filename);
    if (!ifs.is_open()) return false;

    std::string line;
    bool inside_vocab = false;
    while (std::getline(ifs, line)) {
        if (line.find("\"vocab\"") != std::string::npos) {
            inside_vocab = true;
            continue;
        }
        if (inside_vocab) {
            // line 첫 글자가 '}'이면 vocab 종료
            std::string trimmed = line;
            trimmed.erase(trimmed.begin(),
                          std::find_if(trimmed.begin(), trimmed.end(),
                                       [](unsigned char ch) { return !std::isspace(ch); }));
            if (!trimmed.empty() && trimmed[0] == '}') break;

            auto colon = line.rfind(':'); // 마지막 ':' 기준
            if (colon == std::string::npos) continue;

            std::string token = line.substr(0, colon);
            std::string id_str = line.substr(colon + 1);

            // token: 앞뒤 공백, " 제거, 중간 ','는 유지
            token.erase(token.begin(),
                        std::find_if(token.begin(), token.end(),
                                    [](unsigned char ch) { return ch != ' ' && ch != '"'; }));
            token.erase(std::find_if(token.rbegin(), token.rend(),
                                    [](unsigned char ch) { return ch != ' ' && ch != '"'; }).base(),
                        token.end());

            // id: 앞뒤 공백, ',' 제거
            id_str.erase(std::remove(id_str.begin(), id_str.end(), ','), id_str.end());
            id_str.erase(std::remove(id_str.begin(), id_str.end(), ' '), id_str.end());

            if (token.empty() || id_str.empty()) continue;

            int64_t id = std::stoll(id_str);
            vocab[token] = id;
            inv_vocab[id] = token;
        }
    }
    ifs.close();
    return true;
}

// 간단한 SentencePiece 스타일 토크나이저
std::vector<std::string> simple_tokenize(const std::string &text) {
    std::vector<std::string> tokens;
    std::string current;

    for (size_t i = 0; i < text.size(); ++i) {
        unsigned char c = text[i];

        if (c == ' ') {
            if (!current.empty()) {
                tokens.push_back(current);
                current.clear();
            }
            current = "▁";
        } else {
            current += c;
        }
    }

    if (!current.empty()) {
        tokens.push_back(current);
    }
    return tokens;
}

// Decode
std::string decode(const std::vector<int64_t> &ids, std::map<std::string, int64_t> &vocab) {
    std::string result;
    // 역매핑 생성
    std::map<int64_t, std::string> id_to_token;
    for (auto &kv : vocab) id_to_token[kv.second] = kv.first;

    for (auto id : ids) {
        auto it = id_to_token.find(id);
        if (it != id_to_token.end()) {
            std::string tok = it->second;
            if (tok.rfind("▁", 0) == 0) {
                result += " " + tok.substr(3); // "▁" 제거하고 앞에 공백
            } else {
                result += tok;
            }
        }
    }
    return result;
}

void relative_position_bucket_2d(
    std::unique_ptr<int64_t[]>& buckets,
    int query_len,
    int key_len,
    bool bidirectional = true,
    int count = 0,
    int num_buckets = 32,
    int max_distance = 128)
{
    int n = num_buckets;
    if(bidirectional){
        n = n/2;
    }
    int max_exact = n / 2;
    

    for (int q = 0; q < query_len; ++q) {
        for (int k = 0; k < key_len; ++k) {
            int pos = k - q;
            int bucket = 0;
            int r = pos;

            if (bidirectional) {
                if (r > 0)
                    bucket += n;
                r = std::abs(r);
            } else {
                r = -std::min(r-count, 0);
            }

            if (r < max_exact) {
                bucket += r;
            } else {
                float log_ratio = std::log((float)r / max_exact) /
                                   std::log((float)max_distance / max_exact);
                int val = max_exact + static_cast<int>(log_ratio * (n - max_exact));
                val = std::min(val, n - 1);
                bucket += val;
            }

            buckets[q*key_len+k] = static_cast<int64_t>(bucket);
        }
    }
}

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

template <typename T>
std::unique_ptr<T[]> read_bin_file(
    const std::string& filename,
    size_t expected_count 
) {
    std::ifstream file(filename, std::ios::binary);
    if (!file) {
        throw std::runtime_error("파일 열기 실패: " + filename);
    }

    file.seekg(0, std::ios::end);
    std::streamsize file_size = file.tellg();
    file.seekg(0, std::ios::beg);

    const size_t expected_size = expected_count * sizeof(T);
    if (static_cast<size_t>(file_size) != expected_size) {
        throw std::runtime_error(
            "파일 크기가 기대값과 다름. (" +
            std::to_string(file_size) + " vs " +
            std::to_string(expected_size) + ")"
        );
    }

    auto data = std::make_unique<T[]>(expected_count);

    if (!file.read(reinterpret_cast<char*>(data.get()), file_size)) {
        throw std::runtime_error("파일 읽기 실패");
    }

    return data;
}

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

constexpr std::string_view help_message = R"(
Usage: whisper_runner [options] <audio_path>

Required Options:
  -x, --cross_attn_model <path>   Path to the cross-attention model file
  -e, --encoder_model <path>      Path to the encoder model file
  -d, --decoder_model <path>      Path to the decoder model file

Optional:
  -t, --n_thread <int>            Number of threads to use during inference (default: 4)
  -h, --help                      Show this help message and exit

Example:
  optimium-whisper-demo  \
    -t 2 \
    -x models/cross_attn \
    -e models/encoder \
    -d models/decoder \
)";
const size_t required_option = 3;

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

    const char* path_encoder = nullptr;
    const char* path_decoder = nullptr;
    const char* path_cross = nullptr;
    int n_thread = 4;

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
        } else {
            SHOW_HELP_MESSAGE();
        }
        ++i; // skip next (it's the value)
    }

    if (path_encoder == nullptr 
        || path_decoder == nullptr 
        || path_cross == nullptr) {
        SHOW_HELP_MESSAGE();
    }

    // rt::logging::setLogLevel(rt::LogLevel::Debug);
    // rt::logging::addLogWriter(std::make_unique<rt::logging::FileWriter>("rt.log"));

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
    
    std::map<std::string, int64_t> vocab;
    std::map<int64_t, std::string> inv_vocab;
    if (!load_tokenizer_json("tokenizer.json",vocab,inv_vocab)) {
        std::cerr << "Failed to load tokenizer.json\n";
        return 1;
    }

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
    unsigned int n_text_state = 384;
    unsigned int n_text_head = 6;
    unsigned int n_cache_dim = n_text_state*n_text_ctx;
    unsigned int n_text_state_per_head = n_text_state / n_text_head;
    unsigned int n_audio_state = 512;
    unsigned int token_len = 40960;
    unsigned int seqlen = 138;
    
    rt::TypedTensor<uint8_t> weight = rt::tensor<uint8_t>({(token_len*n_audio_state)/(64*2), 64}, 0);
    weight.load("t5_small_embedding_weight.bin");
    rt::TypedTensor<float> weight_scale = rt::tensor<float>({(token_len*n_audio_state)/64, 1}, 0);
    weight_scale.load("t5_small_embedding_weight_scale.bin");
    rt::TypedTensor<float> weight_zp = rt::tensor<float>({(token_len*n_audio_state)/64, 1}, 0);
    weight_zp.load("t5_small_embedding_weight_zp.bin");

    auto t5_start_clock = std::chrono::high_resolution_clock::now();

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
        // std::string input_string = "윤 전 대통령 측은 앞서 소환에는 응하지 않고 방문조사나 서면조사에는 응할 수 있다는 입장을 밝힌 바 가 있는데 오늘 3차 소환에도 여전히 응하지 않고 있는 것으로 보이고 그리고 수사 능력, 다음으로는 지난번에도 말씀드렸듯이 여러 사람들이 같이 모여서 일을 해야 되기 때문에 소통하고 조직에 융합되어야 하기 때문에 그런 부분도 문제가 없는 분이 왔으면 좋겠으며 현재 진행 중인 재판에서 검찰이 증거 인멸 등을 우려해 조건을 달고 풀어주는 보석을 요청했고 법원도 보석을 결정했는데, 김 전 장관 측은 사실상 구속 상태를 불법적으로 연장하는 것 아니냐고 반문했습니다.";
        auto indata = std::make_unique<int64_t[]>(seqlen);
        int64_t indata_value[] = {     1, 32729, 32017, 32008, 30700, 31514, 32641, 29538, 33820, 32048,
         30391, 32089, 32752, 32200, 32636, 32890, 34023, 29695, 32064, 29565,
         34023, 32089, 32752, 29723, 32009, 32773, 35215, 28705, 39957, 38096,
         32083, 32013, 32919, 32687, 28705, 28770, 30734, 32048, 30391, 32260,
         33623, 32752, 32200, 32636, 32063, 32278, 35516, 32277, 34469, 35367,
         28725, 32343, 32863, 32305, 29756, 32260, 32086, 39980, 29470, 37927,
         34636, 32390, 33095, 32659, 34727, 29305, 33252, 33253, 35708, 32291,
         34432, 32035, 32871, 29148, 28705, 39960, 29770, 35148, 33850, 32291,
         32457, 39532, 33376, 32465, 36630, 33491, 32563, 38779, 32097, 32342,
         32400, 34347, 34040, 32005, 38846, 34378, 32031, 38104, 32514, 33858,
         29426, 36388, 32197, 29511, 35610, 32753, 32026, 34653, 33455, 32510,
         32302, 29816, 29599, 32026, 34653, 32662, 33418, 28725, 32146, 32017,
         34457, 32641, 29538, 34803, 34993, 35977, 34300, 32114, 34499, 32022,
         32018, 32205, 36308, 32130, 29710, 32458, 28723, 32000};
        std::copy(std::begin(indata_value), std::end(indata_value), indata.get());
        rt::TypedTensor<int64_t> enc_input = rt::tensor<int64_t>({1, seqlen}, indata_value);
        std::unique_ptr<int64_t[]> encoder_rpe(new int64_t[seqlen*seqlen]);
        relative_position_bucket_2d(encoder_rpe,seqlen,seqlen);
        rt::TypedTensor<int64_t> enc_rpe = rt::tensor<int64_t>({seqlen, seqlen}, encoder_rpe.get());
        std::vector<rt::Tensor> enc_inputs;
        enc_inputs.emplace_back(enc_input);
        enc_inputs.insert(enc_inputs.end(), {weight, weight_scale, weight_zp});
        enc_inputs.emplace_back(enc_rpe);
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
        std::cout << "Execution time: " << duration_ms.count() << " ms\n";
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
        // cross cache 6
        cc60 = std::make_unique<int8_t[]>(n_text_state * seqlen);
        cc61 = std::make_unique<float[]>(n_text_head * seqlen);  
        cc62 = std::make_unique<int8_t[]>(n_text_state * seqlen);
        cc63 = std::make_unique<float[]>(n_text_state);
        cc64 = std::make_unique<int[]>(n_text_state);
        tns_cc60 = rt::tensor<int8_t>({1,n_text_head,seqlen,n_text_state_per_head}, cc60.get());
        tns_cc61 = rt::tensor<float>({1,n_text_head,seqlen,1}, cc61.get());
        tns_cc62 = rt::tensor<int8_t>({1, n_text_head, n_text_state_per_head/16, seqlen, 16}, cc62.get());
        tns_cc63 = rt::tensor<float>({1,n_text_head,1,n_text_state_per_head}, cc63.get()); 
        tns_cc64 = rt::tensor<int>({1,n_text_head,n_text_state_per_head}, cc64.get());
        // cross cache 7
        cc70 = std::make_unique<int8_t[]>(n_text_state * seqlen);
        cc71 = std::make_unique<float[]>(n_text_head * seqlen);  
        cc72 = std::make_unique<int8_t[]>(n_text_state * seqlen);
        cc73 = std::make_unique<float[]>(n_text_state);
        cc74 = std::make_unique<int[]>(n_text_state);
        tns_cc70 = rt::tensor<int8_t>({1,n_text_head,seqlen,n_text_state_per_head}, cc70.get());
        tns_cc71 = rt::tensor<float>({1,n_text_head,seqlen,1}, cc71.get());
        tns_cc72 = rt::tensor<int8_t>({1, n_text_head, n_text_state_per_head/16, seqlen, 16}, cc72.get());
        tns_cc73 = rt::tensor<float>({1,n_text_head,1,n_text_state_per_head}, cc73.get()); 
        tns_cc74 = rt::tensor<int>({1,n_text_head,n_text_state_per_head}, cc74.get());

        std::vector<rt::Tensor> cr_outputs;
        cr_outputs.insert(cr_outputs.end(), {tns_cc00, tns_cc01, tns_cc02, tns_cc03, tns_cc04});
        cr_outputs.insert(cr_outputs.end(), {tns_cc10, tns_cc11, tns_cc12, tns_cc13, tns_cc14});
        cr_outputs.insert(cr_outputs.end(), {tns_cc20, tns_cc21, tns_cc22, tns_cc23, tns_cc24});
        cr_outputs.insert(cr_outputs.end(), {tns_cc30, tns_cc31, tns_cc32, tns_cc33, tns_cc34});
        cr_outputs.insert(cr_outputs.end(), {tns_cc40, tns_cc41, tns_cc42, tns_cc43, tns_cc44});
        cr_outputs.insert(cr_outputs.end(), {tns_cc50, tns_cc51, tns_cc52, tns_cc53, tns_cc54});
        cr_outputs.insert(cr_outputs.end(), {tns_cc60, tns_cc61, tns_cc62, tns_cc63, tns_cc64});
        cr_outputs.insert(cr_outputs.end(), {tns_cc70, tns_cc71, tns_cc72, tns_cc73, tns_cc74});
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
    std::unique_ptr<float[]> logit_out(new float[token_len]);
    
    auto dec_req_start = std::chrono::high_resolution_clock::now();
    rt::InferRequest dec_request = decoder.createRequest();
    rt::TypedTensor<int64_t> idx_dec_input = rt::tensor<int64_t>({1,1}, token_input.get());
    rt::TypedTensor<int64_t> pos_dec_input = rt::tensor<int64_t>({1}, pos_input.get());
    std::unique_ptr<int64_t[]> decoder_rpe(new int64_t[n_text_ctx]);
    rt::TypedTensor<int64_t> dec_rpe = rt::tensor<int64_t>({1, n_text_ctx}, decoder_rpe.get());
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
    
    rt::TypedTensor<float> dec_output = rt::tensor<float>({1,1,token_len}, logit_out.get());
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
    dec_inputs.insert(dec_inputs.end(), {dec_sc60, dec_sc61, tns_cc60, tns_cc61, tns_cc62, tns_cc63, tns_cc64});
    dec_inputs.insert(dec_inputs.end(), {dec_sc70, dec_sc71, tns_cc70, tns_cc71, tns_cc72, tns_cc73, tns_cc74});
    dec_inputs.emplace_back(dec_rpe);
    dec_outputs.emplace_back(dec_output);
    auto dec_req_end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> dec_req_ms = dec_req_end - dec_req_start;
    std::cout << "Decoder Request time : " << dec_req_ms.count() << " ms\n";

    auto dec_loop_start = std::chrono::high_resolution_clock::now();

    int64_t max_i = 2;
    int64_t pos = 0;
    int token_count = 0;
    std::vector<int64_t> result_tensor = {2};
    while(token_count < 400){
        token_input[0] = (int64_t) max_i;
        pos_input[0] = pos;
        relative_position_bucket_2d(decoder_rpe,1,n_text_ctx,false,(int) pos);
        auto decoder_start = std::chrono::high_resolution_clock::now();
        dec_request.infer(rt::make_array(dec_inputs), rt::make_array(dec_outputs));
        dec_request.wait();
        auto decoder_end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> dec_ms = decoder_end - decoder_start;
        std::cout << "Decoder Exe time : " << dec_ms.count() << " ms\n";
        max_i = (int64_t) arg_max(token_len, logit_out.get());
        pos += 1;
        result_tensor.push_back(max_i);
        if(max_i == 32000){
            break;
        }
        token_count++;
    }
    auto dec_loop_end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> dec_loop_ms = dec_loop_end - dec_loop_start;
    std::cout << "Decoder loop Exe time : " << dec_loop_ms.count() << " ms\n";

    for (int64_t x : result_tensor) {
        std::cout << x << " ";
    }
    std::string decode_text = decode(result_tensor,vocab);
    std::cout << "Decoded text: " << decode_text << std::endl;
    auto t5_end_clock = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> t5_clock_ms = t5_end_clock - t5_start_clock;
    std::cout << "t5 clock time: " << t5_clock_ms.count() << " ms\n";
    printf("\nDONE!\n");
    return 0;
}
