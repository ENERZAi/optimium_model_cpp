#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>

// 2D version: returns [query_len][key_len] bucket indices
std::vector<std::vector<int>> relative_position_bucket_2d(
    int query_len,
    int key_len,
    bool bidirectional = true,
    int count = 0,
    int num_buckets = 32,
    int max_distance = 128)
{
    std::vector<std::vector<int>> buckets(query_len, std::vector<int>(key_len));

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

            buckets[q][k] = bucket;
        }
    }

    return buckets;
}

int main() {
    int query_len = 1;
    int key_len = 512;
    
    auto buckets = relative_position_bucket_2d(query_len, key_len, false, 100);

    for (int q = 0; q < query_len; ++q) {
        for (int k = 0; k < key_len; ++k) {
            std::cout << buckets[q][k] << " ";
        }
        std::cout << "\n";
    }

    return 0;
}