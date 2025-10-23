#include <iostream>
#include <fstream>
#include <string>

struct Person {
    int age;
    double height;
    char name[50];
};

int main(){
    Person p1 = {25, 10, "alice"};

    std::ofstream fout("person.bin", std::ios::binary);

    fout.write( (char*) &p1, sizeof(p1));
    fout.close();
};