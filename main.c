#define NN_IMPLEMENTATION
#include "nmat.h"
#include <time.h>


/* ============================================================
 *                     XOR Neural Network
 * ============================================================ */

typedef struct {
    Mat a0;
    Mat w1, b1, a1;
    Mat w2, b2, a2;
} Xor;

float forward(Xor m, float x1, float x2)
{
    /* Input layer */
    mat_at(m.a0, 0, 0) = x1;
    mat_at(m.a0, 0, 1) = x2;


    /* Hidden layer */
    mat_mul(m.a1, m.a0, m.w1);
    mat_sum(m.a1, m.b1);
    mat_sig(m.a1);


    /* Output layer */
    mat_mul(m.a2, m.a1, m.w2);
    mat_sum(m.a2, m.b2);
    mat_sig(m.a2);


    return mat_at(m.a2, 0, 0);
}

float cost(Xor m, Mat ti, Mat to)
{
    NN_ASSERT(ti.rows == to.rows);
    NN_ASSERT(to.cols == 1);

    float c = 0.0f;

    size_t n = ti.rows;
    for (size_t i = 0; i < n; ++i) {
        Mat x = mat_row(ti, i);
        Mat y = mat_row(to, i);

        float prediction = forward(
            m,
            mat_at(x, 0, 0),
            mat_at(x, 0, 1)
        );

        float d = prediction - mat_at(y, 0, 0);
        c += d * d;
    }

    return c / (float)n;
}


/* ============================================================
 *                            Main
 * ============================================================ */
int main(void)
{
    srand((unsigned)time(NULL));
    
    Xor m;

    m.a0 = mat_alloc(1, 2);


    // Hidden Layer
    m.w1 = mat_alloc(2, 2);
    m.b1 = mat_alloc(1, 2);
    m.a1 = mat_alloc(1, 2);


    // Output Layer
    m.w2 = mat_alloc(2, 1);
    m.b2 = mat_alloc(1, 1);
    m.a2 = mat_alloc(1, 1);


    // Random Initialization
    mat_rand(m.w1, -1.f, 1.f);
    mat_rand(m.b1, -1.f, 1.f);

    mat_rand(m.w2, -1.f, 1.f);
    mat_rand(m.b2, -1.f, 1.f);


    /* --------------------------------------------------------
     * XOR Dataset
     *
     * Input:
     *
     * 0 0 -> 0
     * 0 1 -> 1
     * 1 0 -> 1
     * 1 1 -> 0
     *
     * -------------------------------------------------------- */

    Mat ti = mat_alloc(4, 2);
    Mat to = mat_alloc(4, 1);


    /* Inputs */
    mat_at(ti, 0, 0) = 0.f;
    mat_at(ti, 0, 1) = 0.f;

    mat_at(ti, 1, 0) = 0.f;
    mat_at(ti, 1, 1) = 1.f;

    mat_at(ti, 2, 0) = 1.f;
    mat_at(ti, 2, 1) = 0.f;

    mat_at(ti, 3, 0) = 1.f;
    mat_at(ti, 3, 1) = 1.f;


    /* Expected Outputs */
    mat_at(to, 0, 0) = 0.f;
    mat_at(to, 1, 0) = 1.f;
    mat_at(to, 2, 0) = 1.f;
    mat_at(to, 3, 0) = 0.f;


    printf("Initial predictions:\n\n");


    for (size_t i = 0; i < ti.rows; ++i) {
        float x1 = mat_at(ti, i, 0);
        float x2 = mat_at(ti, i, 1);

        float prediction =
            forward(m, x1, x2);

        printf("%.0f XOR %.0f = %f\n", x1, x2, prediction);
    }


    printf("\nInitial cost: %f\n", cost(m, ti, to));

    mat_free(m.a0);

    mat_free(m.w1);
    mat_free(m.b1);
    mat_free(m.a1);

    mat_free(m.w2);
    mat_free(m.b2);
    mat_free(m.a2);

    mat_free(ti);
    mat_free(to);


    return 0;
}