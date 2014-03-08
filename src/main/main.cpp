#include "main.h"

#include "common.h"

int main(int argc, char** argv) {
    if (argc < 3) {
        printf("Usage: %s img1.png img2.png ... imgN.png\n", argv[0]);
        exit(1);
    }

    int imageCount = argc - 1;

    unique_ptr<CImg<float>> prevImg(new CImg<float>(argv[1]));
    unique_ptr<CImg<float>> curImg(new CImg<float>());
    
    for (int imgI = 1; imgI < imageCount; imgI++) {
        *curImg = CImg<float>(argv[1 + imgI]);

        swap(prevImg, curImg);
    }

    return 0;
}

