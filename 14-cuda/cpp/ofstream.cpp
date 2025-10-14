#include <fstream>

int main() {
    std::ofstream output("data2.txt");
    for(int i = 0; i < 10; i++) {
        output << i * i << ' ';
    }
    output << '\n';
    return 0;
}