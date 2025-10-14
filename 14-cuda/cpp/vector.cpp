#include <iostream>
#include <vector>

void print(auto vector) {
    for(auto value : vector) {
        std::cout << value << ' ';
    }
    std::cout << '\n';
}

int main() {
    std::vector<int> my_vector;
    my_vector.reserve(10); //comment this and check how exponentially the memory gets allocated when size is unknown, so reserving is a nicer thing to do
    for(int i = 0; i < 10; i+=1) {
        std::cout << "Size: " << my_vector.size() << '\n';
        std::cout << "Capacity: " << my_vector.capacity() << '\n';
        my_vector.push_back(i);
    }
    return 0;
}