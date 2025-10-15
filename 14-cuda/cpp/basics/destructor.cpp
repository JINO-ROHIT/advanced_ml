#include <iostream>

struct IntArray {
    int *array;
    IntArray(int size) {
        array = new int[size];
    }
    ~IntArray() {
        std::cout << "Running our destructor!\n";
        delete[] array;
    }
};

int main() {
    IntArray a(10);
    a.array[0] = 10;
    std::cout << a.array[0] << '\n';
    return 0;
}