#include <iostream>

struct student {
    int roll;
};


int main(){
    struct student tom = {100};
    struct student *ptr;
    ptr = &tom;
    std::cout << ptr->roll; //arrow operator
    std::cout << (*ptr).roll; //can also dereference first and then use dot operator

}

