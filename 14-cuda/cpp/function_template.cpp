#include <array>
#include <iostream>

void print_array(auto arr){ //using auto makes it work with any iterable
    for(auto value : arr){
        std::cout << value << '\n';
    }
}

int main(){
    std::array<int, 2> arr{1, 2};
    print_array(arr);

    std::array<float, 2> arr2{1.2, 2.4};
    print_array(arr2);
}
