// lets try to do swap using pointers

#include <iostream>

void swap(int *x, int *y){
    int temp = *x;
    *x = *y; // do this first
    *y = temp;
}

int main(){
    int x = 10;
    int y = 20;
    std::cout << "value of x before swap " << x << "value of y before swap" << y;

    swap(&x, &y); // pass by reference

    std::cout << "value of x after swap " << x << "value of y after swap" << y;
    return 0;
}