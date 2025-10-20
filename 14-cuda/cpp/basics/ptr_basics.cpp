#include <iostream>

int main(){
    int x = 10;
    int y = 20;

    int* ptr1 = &x; // &x --> address of x
    int* ptr2 = &y;

    std::cout << "pointer 1 points to " << ptr1;
    std::cout << "x has address " << &x;
    std::cout << "pointer 1 points to value " << *ptr1 ; 
    return 0;
}