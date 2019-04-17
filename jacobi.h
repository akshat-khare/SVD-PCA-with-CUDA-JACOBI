#ifndef JACOBI_H
#define JACOBI_H
#include <iostream>
#include <cmath>
#include <malloc.h>
#include <fstream>

using namespace std;



double** mat_transpose(double** A, int Am, int An);
double** mat_mul(double** A, int Am, int An, 
                 double** B, int Bm, int Bn);
void print_matrix(double** A, int Am, int An);
void print_vector(double* A, int An);
void Jacobi(double **input_matrix, int n, 
            double **eigenvalues, double ***eigenvectors);
#endif