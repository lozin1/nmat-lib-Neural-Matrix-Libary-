/*************************************
*                                     *
*       NN - Small Neural Network     *
*       Matrix Library for C          *
*                                     *
*       STB-style Single Header       *
*                                     *
*************************************/

#ifndef NN_H_
#define NN_H_

#include <stdio.h>
#include <math.h>

#ifndef NN_MALLOC
#include <stdlib.h>
#define NN_MALLOC malloc
#endif

#ifndef NN_FREE
#include <stdlib.h>
#define NN_FREE free
#endif

#ifndef NN_ASSERT
#include <assert.h>
#define NN_ASSERT assert
#endif


/* ============================================================
 *                          Matrix
 * ============================================================ */

typedef struct {
    size_t rows;
    size_t cols;
    float *es;
} Mat;


/*
 * Access an element at row i and column j.
 *
 * Example:
 *
 *     mat_at(m, 0, 1)
 *
 */
#define mat_at(m, i, j) \
    (m).es[(i) * (m).cols + (j)]


/* ============================================================
 *                       Math Functions
 * ============================================================ */

float rand_float(void);
float sigmoidf(float x);


/* ============================================================
 *                      Matrix Functions
 * ============================================================ */

/* Memory */
Mat mat_alloc(size_t rows, size_t cols);
void mat_free(Mat m);


/* Printing */
void mat_print_(Mat m, const char *name);

#define mat_print(m) mat_print_(m, #m)


/* Initialization */
void mat_rand(Mat m, float low, float high);
void mat_fill(Mat m, float x);


/* Matrix Operations */
void mat_sum(Mat dst, Mat a);
void mat_mul(Mat dst, Mat a, Mat b);
void mat_copy(Mat dst, Mat src);


/* Matrix Views */
Mat mat_row(Mat m, size_t row);


/* Activation Functions */
void mat_sig(Mat m);


/* ============================================================
 *                      Implementation
 * ============================================================ */

#ifdef NN_IMPLEMENTATION


/* ============================================================
 *                       Math Functions
 * ============================================================ */

float sigmoidf(float x)
{
    return 1.f / (1.f + expf(-x));
}


float rand_float(void)
{
    return (float) rand() / (float) RAND_MAX;
}


/* ============================================================
 *                      Memory Functions
 * ============================================================ */

Mat mat_alloc(size_t rows, size_t cols)
{
    Mat m;

    m.rows = rows;
    m.cols = cols;

    m.es = NN_MALLOC(sizeof(*m.es) * rows * cols);

    NN_ASSERT(m.es != NULL);

    return m;
}


void mat_free(Mat m)
{
    NN_FREE(m.es);
}


/* ============================================================
 *                       Print Functions
 * ============================================================ */

void mat_print_(Mat m, const char *name)
{
    printf("%s = [\n", name);

    for (size_t i = 0; i < m.rows; ++i) {
        for (size_t j = 0; j < m.cols; ++j) {
            printf("    %f ", mat_at(m, i, j));
        }

        printf("\n");
    }

    printf("]\n");
}


/* ============================================================
 *                   Matrix Initialization
 * ============================================================ */

void mat_rand(Mat m, float low, float high)
{
    for (size_t i = 0; i < m.rows; ++i) {
        for (size_t j = 0; j < m.cols; ++j) {

            mat_at(m, i, j) =
                rand_float() * (high - low) + low;
        }
    }
}


void mat_fill(Mat m, float x)
{
    for (size_t i = 0; i < m.rows; ++i) {
        for (size_t j = 0; j < m.cols; ++j) {

            mat_at(m, i, j) = x;
        }
    }
}


/* ============================================================
 *                    Matrix Operations
 * ============================================================ */

void mat_sum(Mat dst, Mat a)
{
    NN_ASSERT(dst.rows == a.rows);
    NN_ASSERT(dst.cols == a.cols);

    for (size_t i = 0; i < dst.rows; ++i) {
        for (size_t j = 0; j < dst.cols; ++j) {

            mat_at(dst, i, j) += mat_at(a, i, j);
        }
    }
}


void mat_mul(Mat dst, Mat a, Mat b)
{
    NN_ASSERT(a.cols == b.rows);
    NN_ASSERT(dst.rows == a.rows);
    NN_ASSERT(dst.cols == b.cols);

    for (size_t i = 0; i < dst.rows; ++i) {

        for (size_t j = 0; j < dst.cols; ++j) {

            mat_at(dst, i, j) = 0.f;

            for (size_t k = 0; k < a.cols; ++k) {

                /*
                 * C[i][j] += A[i][k] * B[k][j]
                 */

                mat_at(dst, i, j) +=
                    mat_at(a, i, k) *
                    mat_at(b, k, j);
            }
        }
    }
}


/* ============================================================
 *                      Matrix Copy
 * ============================================================ */

void mat_copy(Mat dst, Mat src)
{
    NN_ASSERT(dst.rows == src.rows);
    NN_ASSERT(dst.cols == src.cols);

    for (size_t i = 0; i < dst.rows; ++i) {
        for (size_t j = 0; j < dst.cols; ++j) {

            mat_at(dst, i, j) =
                mat_at(src, i, j);
        }
    }
}


/* ============================================================
 *                       Matrix Views
 * ============================================================ */

/*
 * Returns a view of a single row.
 *
 * IMPORTANT:
 *
 * The returned Mat does not own its memory.
 * Do not call mat_free() on the returned row.
 */

Mat mat_row(Mat m, size_t row)
{
    NN_ASSERT(row < m.rows);

    return (Mat) {
        .rows = 1,
        .cols = m.cols,
        .es = &mat_at(m, row, 0),
    };
}


/* ============================================================
 *                    Activation Functions
 * ============================================================ */

void mat_sig(Mat m)
{
    for (size_t i = 0; i < m.rows; ++i) {
        for (size_t j = 0; j < m.cols; ++j) {

            mat_at(m, i, j) =
                sigmoidf(mat_at(m, i, j));
        }
    }
}


#endif /* NN_IMPLEMENTATION */

#endif /* NN_H_ */