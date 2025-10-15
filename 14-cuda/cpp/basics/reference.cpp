#include <iostream>

int main() {
    int a = 5;
    int b = a;
    b += 1;

    int c = 5;
    int *d = &c; //points to address of c
    *d += 1; //derefrence and increment c by 1


    std::cout << "a = " << &a << '\n';
    std::cout << "b = " << &b << '\n';

    //both have the same address
    std::cout << "c = " << &c << '\n';
    std::cout << "d = " << d << '\n';
    return 0;
}