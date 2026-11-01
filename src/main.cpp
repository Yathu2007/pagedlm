// pagedlm: entry point.
// M0 TODO (yours): mmap the .bin, validate the header and index, print every tensor.
// The on-disk format is specified in docs/format.md.

#include <cstdio>

int main(int argc, char** argv) {
    if (argc < 2) {
        std::fprintf(stderr, "usage: %s <model.bin>\n", argv[0]);
        return 1;
    }
    std::fprintf(stderr, "loader not implemented yet\n");
    return 1;
}
