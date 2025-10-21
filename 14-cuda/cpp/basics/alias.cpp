#include <stdio.h>

typedef int Integer;

int main() {
  
    // n is of type int, but we are using
  	// alias Integer
    Integer n = 10;
  
    printf("%d", n);
    return 0;
}