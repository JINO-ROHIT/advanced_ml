#include <iostream>
#include <fstream>
#include <string>

struct Person {
    int age;
    double height;
    char name[50];
};

int main() {
    Person p2;

    std::ifstream finp("person.bin", std::ios::binary);

    finp.read((char*) &p2, sizeof(p2));
    finp.close();

    std::cout << "Name: " << p2.name << "\n";
    std::cout << "Age: " << p2.age << "\n";
    std::cout << "Height: " << p2.height << "\n";

    return 0;
}
