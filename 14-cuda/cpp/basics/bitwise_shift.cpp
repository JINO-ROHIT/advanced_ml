#include <iostream>

int main(){
    // 1        = 0000 0000 0000 0001
    // 1 << 16  = 0001 0000 0000 0000 0000
    // 2^16 = 65536
    static __fp16 table_gelu_f16[1 << 16];
    size_t num_elements = sizeof(table_gelu_f16) / sizeof(table_gelu_f16[0]);
    size_t total_bytes = sizeof(table_gelu_f16);

    printf("Number of elements: %zu\n", num_elements);
    printf("Total size in bytes: %zu\n", total_bytes);
}