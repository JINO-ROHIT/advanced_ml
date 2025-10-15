#include <iostream>

struct Point {
    int x;
    int y;

    void print() {
        std::cout << "x = " << x << '\n';
        std::cout << "y = " << y << '\n';
    }
};

int main() {
    Point p1;
    p1.x = 10;
    p1.y = 20;
    p1.print();

    Point p2;
    p2.x = 5;
    p2.y = -5;
    p2.print();
    return 0;
}