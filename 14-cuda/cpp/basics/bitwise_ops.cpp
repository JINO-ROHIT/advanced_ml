#include <stdio.h>

int main() {
    
    // a = 5 (0000 0101 in 8-bit binary) 
    // b = 9 (0000 1001 in 8-bit binary)
    unsigned int a = 5, b = 9;

    // The result is 0000 0001
    printf("a&b = %u\n", a & b);

    // The result is 0000 1101
    printf("a|b = %u\n", a | b);

    // The result is 0000 1100
    printf("a^b = %u\n", a ^ b);

    // The result is 11111111111111111111111111111010
    // (assuming 32-bit unsigned int)
    printf("~a = %u\n", a = ~a);

    // The result is 00010010
    printf("b<<1 = %u\n", b << 1);

    // The result is 00000100
    printf("b>>1 = %u\n", b >> 1);
    return 0;
}