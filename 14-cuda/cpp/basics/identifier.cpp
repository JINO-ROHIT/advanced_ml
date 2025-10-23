#include <cstdio>

void load_model() {
    printf("%s: starting model load...\n", __func__); // the __func__ auto replace by the function name
    printf("%s: model loaded successfully!\n", __func__);
}

int main() {
    load_model();
    return 0;
}