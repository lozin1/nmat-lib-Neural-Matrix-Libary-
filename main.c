#define NN_IMPLEMENTATION
#include "nmat.h"
#include <time.h>

/* Dataset: AND Gate */
float td_and[] = {
    0, 0, 0,
    1, 0, 0,
    0, 1, 0,
    1, 1, 1,
};

/* Dataset: NAND Gate */
float td_nand[] = {
    0, 0, 1,
    1, 0, 1,
    0, 1, 1,
    1, 1, 0,
};

/* Dataset: OR Gate */
float td_or[] = {
    0, 0, 0,
    1, 0, 1,
    0, 1, 1,
    1, 1, 1,
};

/* Dataset: XOR Gate */
float td_xor[] = {
    0, 0, 1,
    1, 0, 1,
    0, 1, 1,
    1, 1, 0,
};

int main(void)
{
    srand((unsigned)time(NULL));

    float *td = td_xor;

    size_t stride = 3;
    size_t n = 4;

    Mat ti = {
        .rows = n,
        .cols = 2,
        .stride = stride,
        .es = td,
    };

    Mat to = {
        .rows = n,
        .cols = 1,
        .stride = stride,
        .es = td + 2,
    };

    // تعريف المعمارية: 2 مدخلات -> 2 نيورون مخفي -> 1 مخرج
    size_t arch[] = {2, 2, 1};
    
    NN nn = nn_alloc(arch, ARRAY_LEN(arch));
    NN g  = nn_alloc(arch, ARRAY_LEN(arch)); // Gradient NN

    nn_rand(nn, -1.f, 1.f);

    printf("Cost before training: %f\n\n", nn_cost(nn, ti, to));

    // حلقة التدريب (Training Loop)
    float eps = 1e-1f;
    float rate = 1e-1f;

    for (size_t i = 0; i < 100000; ++i) {
        nn_finite_diff(nn, g, eps, ti, to);
        nn_learn(nn, g, rate);
    }

    printf("Cost after training: %f\n\n", nn_cost(nn, ti, to));

    // طباعة التوقعات بعد التدريب
    printf("Predictions:\n");
    for (size_t i = 0; i < ti.rows; ++i) {
        Mat x = mat_row(ti, i);
        mat_copy(NN_INPUT(nn), x);
        nn_forward(nn);

        float x1 = mat_at(ti, i, 0);
        float x2 = mat_at(ti, i, 1);
        float y  = mat_at(NN_OUTPUT(nn), 0, 0);

        printf("%.0f &/ %.0f = %f\n", x1, x2, y);
    }

    // تنظيف الذاكرة
    nn_free(nn);
    nn_free(g);

    return 0;
}