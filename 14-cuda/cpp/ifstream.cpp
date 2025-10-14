#include <fstream>
#include <iostream>
#include <string>

int main() {
    std::ifstream input("data.txt"); // read from data.txt
    if (!input.is_open()) {
        std::cerr << "Failed to open file!\n";
        return 1;
    }

    std::string line;
    while (std::getline(input, line)) {
        std::cout << line << '\n';
    }
}
