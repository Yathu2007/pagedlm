#include <cstdio>

int main(int argc, char** argv) {
    if (argc < 2) {
        std::fprintf(stderr, "usage: %s <model.bin>\n", argv[0]);
        return 1;
    }
    std::fprintf(stderr, "loader not implemented yet\n");
    return 1;
}
