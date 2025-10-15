## Run in ubuntu with cmake
```
cd build && make && cd ..
./build/optimium-whisper-demo -m models/whisper -t 2 -x models/cross_attn \
    -e models/encoder -d models/decoder sample.wav
```

## Run in yocto without cmake but with g++
```
g++ -I <Optimium runtime directory>/include/Optimium-Runtime -L <Optimium runtime directory>/linux-arm64/lib \
  -loptimium-runtime -o optimium-whisper-demo main.cpp melspecto.cpp token.cpp loader.cpp common-whisper.cpp sampling.cpp

LD_LIBRARY_PATH=<Optimium runtime directory>/linux-arm64/lib ./optimium-whisper-demo -m models/whisper -t 2 -x models/cross_attn \
    -e models/encoder -d models/decoder sample.wav
```

