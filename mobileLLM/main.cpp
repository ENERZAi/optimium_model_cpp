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

int main(int argc, char ** argv){
    // Optimium runtime setting
    rt::logging::setLogLevel(rt::LogLevel::Error);
    // rt::logging::addLogWriter(std::make_unique<rt::logging::FileWriter>("rt.log"));

    rt::AutoInit Init;

    std::cout << "load start" << std::endl;
    rt::config::setPrintThreshold(-1);
    
    rt::ModelLoadOptions DecOptions;
    DecOptions.ThreadCount = 4;
    DecOptions.DisableDenormals = false;

    unsigned int n_text_head = 6; 
    unsigned int n_text_ctx = 64;
    unsigned int n_text_state_per_head= 64;
    unsigned int n_cache_dim = n_text_head*n_text_ctx*n_text_state_per_head;

    // TODO: get inputs_id dynamic
    int64_t inputs_id[] = {     1,   910,   338,   534,  1161, 29889,   577};
    int num_inputs_id = sizeof(inputs_id) / sizeof(int64_t );
    std::vector<int64_t> result_tensor = {   1,   910,   338,   534,  1161, 29889,   577};
    auto dec_load_start = std::chrono::high_resolution_clock::now();
    rt::Model decoder = rt::loadModel("/home/root/mobileLLM/mobileLLM_600M", DecOptions);
    auto dec_load_end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> dec_load_ms = dec_load_end - dec_load_start;
    std::cout << "decoder loaded" << std::endl;
    std::cout << "Decoder load time: " << dec_load_ms.count() << " ms\n";

    std::unique_ptr<int64_t[]> token_input(new int64_t[1]);
    std::unique_ptr<int64_t[]> pos_input(new int64_t[1]);
    std::unique_ptr<float[]> logit_out(new float[32000]);
    
    auto dec_req_start = std::chrono::high_resolution_clock::now();
    rt::InferRequest dec_request = decoder.createRequest();
    rt::TypedTensor<int64_t> idx_dec_input = rt::tensor<int64_t>({1,1}, token_input.get());
    rt::TypedTensor<int64_t> pos_dec_input = rt::tensor<int64_t>({1,1}, pos_input.get());
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

    // self kv cache - 12
    auto sc120 = std::make_unique<rt::float16[]>(n_cache_dim);
    auto sc121 = std::make_unique<rt::float16[]>(n_cache_dim);
    rt::TypedTensor<rt::float16> dec_sc120 = rt::tensor<rt::float16>({1,n_text_head,n_text_ctx,n_text_state_per_head}, sc120.get());
    rt::TypedTensor<rt::float16> dec_sc121 = rt::tensor<rt::float16>({1,n_text_head,n_text_ctx,n_text_state_per_head}, sc121.get());

    // self kv cache - 13
    auto sc130 = std::make_unique<rt::float16[]>(n_cache_dim);
    auto sc131 = std::make_unique<rt::float16[]>(n_cache_dim);
    rt::TypedTensor<rt::float16> dec_sc130 = rt::tensor<rt::float16>({1,n_text_head,n_text_ctx,n_text_state_per_head}, sc130.get());
    rt::TypedTensor<rt::float16> dec_sc131 = rt::tensor<rt::float16>({1,n_text_head,n_text_ctx,n_text_state_per_head}, sc131.get());

    // self kv cache - 14
    auto sc140 = std::make_unique<rt::float16[]>(n_cache_dim);
    auto sc141 = std::make_unique<rt::float16[]>(n_cache_dim);
    rt::TypedTensor<rt::float16> dec_sc140 = rt::tensor<rt::float16>({1,n_text_head,n_text_ctx,n_text_state_per_head}, sc140.get());
    rt::TypedTensor<rt::float16> dec_sc141 = rt::tensor<rt::float16>({1,n_text_head,n_text_ctx,n_text_state_per_head}, sc141.get());

    // self kv cache - 15
    auto sc150 = std::make_unique<rt::float16[]>(n_cache_dim);
    auto sc151 = std::make_unique<rt::float16[]>(n_cache_dim);
    rt::TypedTensor<rt::float16> dec_sc150 = rt::tensor<rt::float16>({1,n_text_head,n_text_ctx,n_text_state_per_head}, sc150.get());
    rt::TypedTensor<rt::float16> dec_sc151 = rt::tensor<rt::float16>({1,n_text_head,n_text_ctx,n_text_state_per_head}, sc151.get());

    // self kv cache - 16
    auto sc160 = std::make_unique<rt::float16[]>(n_cache_dim);
    auto sc161 = std::make_unique<rt::float16[]>(n_cache_dim);
    rt::TypedTensor<rt::float16> dec_sc160 = rt::tensor<rt::float16>({1,n_text_head,n_text_ctx,n_text_state_per_head}, sc160.get());
    rt::TypedTensor<rt::float16> dec_sc161 = rt::tensor<rt::float16>({1,n_text_head,n_text_ctx,n_text_state_per_head}, sc161.get());

    // self kv cache - 17
    auto sc170 = std::make_unique<rt::float16[]>(n_cache_dim);
    auto sc171 = std::make_unique<rt::float16[]>(n_cache_dim);
    rt::TypedTensor<rt::float16> dec_sc170 = rt::tensor<rt::float16>({1,n_text_head,n_text_ctx,n_text_state_per_head}, sc170.get());
    rt::TypedTensor<rt::float16> dec_sc171 = rt::tensor<rt::float16>({1,n_text_head,n_text_ctx,n_text_state_per_head}, sc171.get());

    // self kv cache - 18
    auto sc180 = std::make_unique<rt::float16[]>(n_cache_dim);
    auto sc181 = std::make_unique<rt::float16[]>(n_cache_dim);
    rt::TypedTensor<rt::float16> dec_sc180 = rt::tensor<rt::float16>({1,n_text_head,n_text_ctx,n_text_state_per_head}, sc180.get());
    rt::TypedTensor<rt::float16> dec_sc181 = rt::tensor<rt::float16>({1,n_text_head,n_text_ctx,n_text_state_per_head}, sc181.get());

    // self kv cache - 19
    auto sc190 = std::make_unique<rt::float16[]>(n_cache_dim);
    auto sc191 = std::make_unique<rt::float16[]>(n_cache_dim);
    rt::TypedTensor<rt::float16> dec_sc190 = rt::tensor<rt::float16>({1,n_text_head,n_text_ctx,n_text_state_per_head}, sc190.get());
    rt::TypedTensor<rt::float16> dec_sc191 = rt::tensor<rt::float16>({1,n_text_head,n_text_ctx,n_text_state_per_head}, sc191.get());
    
    // self kv cache - 20
    auto sc200 = std::make_unique<rt::float16[]>(n_cache_dim);
    auto sc201 = std::make_unique<rt::float16[]>(n_cache_dim);
    rt::TypedTensor<rt::float16> dec_sc200 = rt::tensor<rt::float16>({1,n_text_head,n_text_ctx,n_text_state_per_head}, sc200.get());
    rt::TypedTensor<rt::float16> dec_sc201 = rt::tensor<rt::float16>({1,n_text_head,n_text_ctx,n_text_state_per_head}, sc201.get());
    
    // self kv cache - 21
    auto sc210 = std::make_unique<rt::float16[]>(n_cache_dim);
    auto sc211 = std::make_unique<rt::float16[]>(n_cache_dim);
    rt::TypedTensor<rt::float16> dec_sc210 = rt::tensor<rt::float16>({1,n_text_head,n_text_ctx,n_text_state_per_head}, sc210.get());
    rt::TypedTensor<rt::float16> dec_sc211 = rt::tensor<rt::float16>({1,n_text_head,n_text_ctx,n_text_state_per_head}, sc211.get());
    
    // self kv cache - 22
    auto sc220 = std::make_unique<rt::float16[]>(n_cache_dim);
    auto sc221 = std::make_unique<rt::float16[]>(n_cache_dim);
    rt::TypedTensor<rt::float16> dec_sc220 = rt::tensor<rt::float16>({1,n_text_head,n_text_ctx,n_text_state_per_head}, sc220.get());
    rt::TypedTensor<rt::float16> dec_sc221 = rt::tensor<rt::float16>({1,n_text_head,n_text_ctx,n_text_state_per_head}, sc221.get());
    
    // self kv cache - 23
    auto sc230 = std::make_unique<rt::float16[]>(n_cache_dim);
    auto sc231 = std::make_unique<rt::float16[]>(n_cache_dim);
    rt::TypedTensor<rt::float16> dec_sc230 = rt::tensor<rt::float16>({1,n_text_head,n_text_ctx,n_text_state_per_head}, sc230.get());
    rt::TypedTensor<rt::float16> dec_sc231 = rt::tensor<rt::float16>({1,n_text_head,n_text_ctx,n_text_state_per_head}, sc231.get());
    
    // self kv cache - 24
    auto sc240 = std::make_unique<rt::float16[]>(n_cache_dim);
    auto sc241 = std::make_unique<rt::float16[]>(n_cache_dim);
    rt::TypedTensor<rt::float16> dec_sc240 = rt::tensor<rt::float16>({1,n_text_head,n_text_ctx,n_text_state_per_head}, sc240.get());
    rt::TypedTensor<rt::float16> dec_sc241 = rt::tensor<rt::float16>({1,n_text_head,n_text_ctx,n_text_state_per_head}, sc241.get());
    
    // self kv cache - 25
    auto sc250 = std::make_unique<rt::float16[]>(n_cache_dim);
    auto sc251 = std::make_unique<rt::float16[]>(n_cache_dim);
    rt::TypedTensor<rt::float16> dec_sc250 = rt::tensor<rt::float16>({1,n_text_head,n_text_ctx,n_text_state_per_head}, sc250.get());
    rt::TypedTensor<rt::float16> dec_sc251 = rt::tensor<rt::float16>({1,n_text_head,n_text_ctx,n_text_state_per_head}, sc251.get());
    
    // self kv cache - 26
    auto sc260 = std::make_unique<rt::float16[]>(n_cache_dim);
    auto sc261 = std::make_unique<rt::float16[]>(n_cache_dim);
    rt::TypedTensor<rt::float16> dec_sc260 = rt::tensor<rt::float16>({1,n_text_head,n_text_ctx,n_text_state_per_head}, sc260.get());
    rt::TypedTensor<rt::float16> dec_sc261 = rt::tensor<rt::float16>({1,n_text_head,n_text_ctx,n_text_state_per_head}, sc261.get());
    
    // self kv cache - 27
    auto sc270 = std::make_unique<rt::float16[]>(n_cache_dim);
    auto sc271 = std::make_unique<rt::float16[]>(n_cache_dim);
    rt::TypedTensor<rt::float16> dec_sc270 = rt::tensor<rt::float16>({1,n_text_head,n_text_ctx,n_text_state_per_head}, sc270.get());
    rt::TypedTensor<rt::float16> dec_sc271 = rt::tensor<rt::float16>({1,n_text_head,n_text_ctx,n_text_state_per_head}, sc271.get());
    
    // self kv cache - 28
    auto sc280 = std::make_unique<rt::float16[]>(n_cache_dim);
    auto sc281 = std::make_unique<rt::float16[]>(n_cache_dim);
    rt::TypedTensor<rt::float16> dec_sc280 = rt::tensor<rt::float16>({1,n_text_head,n_text_ctx,n_text_state_per_head}, sc280.get());
    rt::TypedTensor<rt::float16> dec_sc281 = rt::tensor<rt::float16>({1,n_text_head,n_text_ctx,n_text_state_per_head}, sc281.get());
    
    // self kv cache - 29
    auto sc290 = std::make_unique<rt::float16[]>(n_cache_dim);
    auto sc291 = std::make_unique<rt::float16[]>(n_cache_dim);
    rt::TypedTensor<rt::float16> dec_sc290 = rt::tensor<rt::float16>({1,n_text_head,n_text_ctx,n_text_state_per_head}, sc290.get());
    rt::TypedTensor<rt::float16> dec_sc291 = rt::tensor<rt::float16>({1,n_text_head,n_text_ctx,n_text_state_per_head}, sc291.get());
    
    // self kv cache - 30
    auto sc300 = std::make_unique<rt::float16[]>(n_cache_dim);
    auto sc301 = std::make_unique<rt::float16[]>(n_cache_dim);
    rt::TypedTensor<rt::float16> dec_sc300 = rt::tensor<rt::float16>({1,n_text_head,n_text_ctx,n_text_state_per_head}, sc300.get());
    rt::TypedTensor<rt::float16> dec_sc301 = rt::tensor<rt::float16>({1,n_text_head,n_text_ctx,n_text_state_per_head}, sc301.get());
    
    // self kv cache - 31
    auto sc310 = std::make_unique<rt::float16[]>(n_cache_dim);
    auto sc311 = std::make_unique<rt::float16[]>(n_cache_dim);
    rt::TypedTensor<rt::float16> dec_sc310 = rt::tensor<rt::float16>({1,n_text_head,n_text_ctx,n_text_state_per_head}, sc310.get());
    rt::TypedTensor<rt::float16> dec_sc311 = rt::tensor<rt::float16>({1,n_text_head,n_text_ctx,n_text_state_per_head}, sc311.get());
    
    // self kv cache - 32
    auto sc320 = std::make_unique<rt::float16[]>(n_cache_dim);
    auto sc321 = std::make_unique<rt::float16[]>(n_cache_dim);
    rt::TypedTensor<rt::float16> dec_sc320 = rt::tensor<rt::float16>({1,n_text_head,n_text_ctx,n_text_state_per_head}, sc320.get());
    rt::TypedTensor<rt::float16> dec_sc321 = rt::tensor<rt::float16>({1,n_text_head,n_text_ctx,n_text_state_per_head}, sc321.get());
    
    // self kv cache - 33
    auto sc330 = std::make_unique<rt::float16[]>(n_cache_dim);
    auto sc331 = std::make_unique<rt::float16[]>(n_cache_dim);
    rt::TypedTensor<rt::float16> dec_sc330 = rt::tensor<rt::float16>({1,n_text_head,n_text_ctx,n_text_state_per_head}, sc330.get());
    rt::TypedTensor<rt::float16> dec_sc331 = rt::tensor<rt::float16>({1,n_text_head,n_text_ctx,n_text_state_per_head}, sc331.get());
    
    // self kv cache - 34
    auto sc340 = std::make_unique<rt::float16[]>(n_cache_dim);
    auto sc341 = std::make_unique<rt::float16[]>(n_cache_dim);
    rt::TypedTensor<rt::float16> dec_sc340 = rt::tensor<rt::float16>({1,n_text_head,n_text_ctx,n_text_state_per_head}, sc340.get());
    rt::TypedTensor<rt::float16> dec_sc341 = rt::tensor<rt::float16>({1,n_text_head,n_text_ctx,n_text_state_per_head}, sc341.get());
    
    // self kv cache - 35
    auto sc350 = std::make_unique<rt::float16[]>(n_cache_dim);
    auto sc351 = std::make_unique<rt::float16[]>(n_cache_dim);
    rt::TypedTensor<rt::float16> dec_sc350 = rt::tensor<rt::float16>({1,n_text_head,n_text_ctx,n_text_state_per_head}, sc350.get());
    rt::TypedTensor<rt::float16> dec_sc351 = rt::tensor<rt::float16>({1,n_text_head,n_text_ctx,n_text_state_per_head}, sc351.get());
    
    // self kv cache - 36
    auto sc360 = std::make_unique<rt::float16[]>(n_cache_dim);
    auto sc361 = std::make_unique<rt::float16[]>(n_cache_dim);
    rt::TypedTensor<rt::float16> dec_sc360 = rt::tensor<rt::float16>({1,n_text_head,n_text_ctx,n_text_state_per_head}, sc360.get());
    rt::TypedTensor<rt::float16> dec_sc361 = rt::tensor<rt::float16>({1,n_text_head,n_text_ctx,n_text_state_per_head}, sc361.get());
    
    // self kv cache - 37
    auto sc370 = std::make_unique<rt::float16[]>(n_cache_dim);
    auto sc371 = std::make_unique<rt::float16[]>(n_cache_dim);
    rt::TypedTensor<rt::float16> dec_sc370 = rt::tensor<rt::float16>({1,n_text_head,n_text_ctx,n_text_state_per_head}, sc370.get());
    rt::TypedTensor<rt::float16> dec_sc371 = rt::tensor<rt::float16>({1,n_text_head,n_text_ctx,n_text_state_per_head}, sc371.get());
    
    // self kv cache - 38
    auto sc380 = std::make_unique<rt::float16[]>(n_cache_dim);
    auto sc381 = std::make_unique<rt::float16[]>(n_cache_dim);
    rt::TypedTensor<rt::float16> dec_sc380 = rt::tensor<rt::float16>({1,n_text_head,n_text_ctx,n_text_state_per_head}, sc380.get());
    rt::TypedTensor<rt::float16> dec_sc381 = rt::tensor<rt::float16>({1,n_text_head,n_text_ctx,n_text_state_per_head}, sc381.get());
    
    // self kv cache - 39
    auto sc390 = std::make_unique<rt::float16[]>(n_cache_dim);
    auto sc391 = std::make_unique<rt::float16[]>(n_cache_dim);
    rt::TypedTensor<rt::float16> dec_sc390 = rt::tensor<rt::float16>({1,n_text_head,n_text_ctx,n_text_state_per_head}, sc390.get());
    rt::TypedTensor<rt::float16> dec_sc391 = rt::tensor<rt::float16>({1,n_text_head,n_text_ctx,n_text_state_per_head}, sc391.get());
    
    rt::TypedTensor<float> dec_output = rt::tensor<float>({1,1,32000}, logit_out.get());
    std::vector<rt::Tensor> dec_outputs;
    dec_inputs.emplace_back(idx_dec_input);
    dec_inputs.emplace_back(pos_dec_input);
    dec_inputs.insert(dec_inputs.end(), {dec_sc00, dec_sc01});
    dec_inputs.insert(dec_inputs.end(), {dec_sc10, dec_sc11});
    dec_inputs.insert(dec_inputs.end(), {dec_sc20, dec_sc21});
    dec_inputs.insert(dec_inputs.end(), {dec_sc30, dec_sc31});
    dec_inputs.insert(dec_inputs.end(), {dec_sc40, dec_sc41});
    dec_inputs.insert(dec_inputs.end(), {dec_sc50, dec_sc51});
    dec_inputs.insert(dec_inputs.end(), {dec_sc60, dec_sc61});
    dec_inputs.insert(dec_inputs.end(), {dec_sc70, dec_sc71});
    dec_inputs.insert(dec_inputs.end(), {dec_sc80, dec_sc81});
    dec_inputs.insert(dec_inputs.end(), {dec_sc90, dec_sc91});

    dec_inputs.insert(dec_inputs.end(), {dec_sc100, dec_sc101});
    dec_inputs.insert(dec_inputs.end(), {dec_sc110, dec_sc111});
    dec_inputs.insert(dec_inputs.end(), {dec_sc120, dec_sc121});
    dec_inputs.insert(dec_inputs.end(), {dec_sc130, dec_sc131});
    dec_inputs.insert(dec_inputs.end(), {dec_sc140, dec_sc141});
    dec_inputs.insert(dec_inputs.end(), {dec_sc150, dec_sc151});
    dec_inputs.insert(dec_inputs.end(), {dec_sc160, dec_sc161});
    dec_inputs.insert(dec_inputs.end(), {dec_sc170, dec_sc171});
    dec_inputs.insert(dec_inputs.end(), {dec_sc180, dec_sc181});
    dec_inputs.insert(dec_inputs.end(), {dec_sc190, dec_sc191});

    dec_inputs.insert(dec_inputs.end(), {dec_sc200, dec_sc201});
    dec_inputs.insert(dec_inputs.end(), {dec_sc210, dec_sc211});
    dec_inputs.insert(dec_inputs.end(), {dec_sc220, dec_sc221});
    dec_inputs.insert(dec_inputs.end(), {dec_sc230, dec_sc231});
    dec_inputs.insert(dec_inputs.end(), {dec_sc240, dec_sc241});
    dec_inputs.insert(dec_inputs.end(), {dec_sc250, dec_sc251});
    dec_inputs.insert(dec_inputs.end(), {dec_sc260, dec_sc261});
    dec_inputs.insert(dec_inputs.end(), {dec_sc270, dec_sc271});
    dec_inputs.insert(dec_inputs.end(), {dec_sc280, dec_sc281});
    dec_inputs.insert(dec_inputs.end(), {dec_sc290, dec_sc291});

    dec_inputs.insert(dec_inputs.end(), {dec_sc300, dec_sc301});
    dec_inputs.insert(dec_inputs.end(), {dec_sc310, dec_sc311});
    dec_inputs.insert(dec_inputs.end(), {dec_sc320, dec_sc321});
    dec_inputs.insert(dec_inputs.end(), {dec_sc330, dec_sc331});
    dec_inputs.insert(dec_inputs.end(), {dec_sc340, dec_sc341});
    dec_inputs.insert(dec_inputs.end(), {dec_sc350, dec_sc351});
    dec_inputs.insert(dec_inputs.end(), {dec_sc360, dec_sc361});
    dec_inputs.insert(dec_inputs.end(), {dec_sc370, dec_sc371});
    dec_inputs.insert(dec_inputs.end(), {dec_sc380, dec_sc381});
    dec_inputs.insert(dec_inputs.end(), {dec_sc390, dec_sc391});
    dec_outputs.emplace_back(dec_output);

    for(int i = 0; i < num_inputs_id; i++){
        token_input[0] = inputs_id[i];
        pos_input[0] = (int64_t) i;
        auto decoder_start = std::chrono::high_resolution_clock::now();
        dec_request.infer(rt::make_array(dec_inputs), rt::make_array(dec_outputs));
        dec_request.wait();
        auto decoder_end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> dec_ms = decoder_end - decoder_start;
        std::cout << "Decoder Exe time : " << dec_ms.count() << " ms\n";
    }
    int max_i = arg_max(32000, logit_out.get());
    result_tensor.push_back((int64_t) max_i);
    for(int i = num_inputs_id; i < num_inputs_id+50; i++){
        token_input[0] = (int64_t) max_i;
        pos_input[0] = (int64_t) i;
        auto decoder_start = std::chrono::high_resolution_clock::now();
        dec_request.infer(rt::make_array(dec_inputs), rt::make_array(dec_outputs));
        dec_request.wait();
        auto decoder_end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> dec_ms = decoder_end - decoder_start;
        std::cout << "Decoder Exe time : " << dec_ms.count() << " ms\n";

        max_i = arg_max(32000, logit_out.get());
        result_tensor.push_back((int64_t) max_i);
    }
    for (int64_t x : result_tensor) {
        std::cout << x << " ";
    }
    std::cout << "\n";
    return 0;
}