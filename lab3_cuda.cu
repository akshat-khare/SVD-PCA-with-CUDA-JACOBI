#include "lab3_cuda.h"
#include <iostream>
#include <cmath>
#include <fstream>
using namespace std;
// /*
// 	*****************************************************
// 		TODO -- You must implement this function
// 	*****************************************************
// */

#define TOLERANCE 0.001
#define JACOBI_UPDATE_TOLERANCE 0.001
// #define FILENAME "testcases/pdftestcase1"
int samples;
int features;

double **S; //Symmetric matrix (input)
double  *e; //eigenvalues
double **E; //eigenvectors
int  *ind;
bool *changed;
int  state;
int  N_jacobi;

void read_file(char* filename, int num_samples, int num_features, double** A) {
    ifstream ifile;
    ifile.open(filename, ios::in);

    double tmp;
    for (int i=0; i<num_samples; i++) {
        for (int j=0; j<num_features; j++){
            ifile >> tmp;
            // cout <<tmp;
            A[i][j] = tmp;
        }
    }

    ifile.close();
}

double** mat_transpose(double** A, int Am, int An) {
    double **B;
    B = (double**)malloc(__SIZEOF_POINTER__*An);
    for (int i=0; i<An; i++)
        B[i] = (double*)malloc(__SIZEOF_DOUBLE__*Am);

    for (int i=0; i<Am; i++){
        for (int j=0; j<An; j++){
            B[j][i] = A[i][j];
        }
    }

    return B;
}

double** mat_mul(double** A, int Am, int An, 
                 double** B, int Bm, int Bn){
    double **C;
    C = (double**)malloc(__SIZEOF_POINTER__*Am);
    for (int i=0; i<Am; i++)
        C[i] = (double*)malloc(__SIZEOF_DOUBLE__*Bn);

    for (int i=0; i<Am; i++){
        for (int j=0; j<Bn; j++){
            C[i][j] = 0;
            for (int k=0; k<An; k++){
                C[i][j] += A[i][k] * B[k][j];
            }
        }
    }

    return C;
}

int maxind(int k) {
    int m = k+1;

    for (int i = k+2; i < N_jacobi; i++){
        if (fabs(S[k][i]) > fabs(S[k][m])){
            m = i;
        }
    }

    return m;
}

void update(int k, double t) {
    double ek_prev = e[k];
    e[k] = ek_prev + t;

    if (e[k] < 0) e[k] = 0;

    if (changed[k] && fabs(ek_prev - e[k]) < JACOBI_UPDATE_TOLERANCE) {
        changed[k] = false;
        state = state - 1;
    }
    else if ((! changed[k]) && fabs(ek_prev - e[k]) > JACOBI_UPDATE_TOLERANCE) {
        changed[k] = true;
        state = state + 1;
    }
}

void rotate(int k, int l, int i, int j, double c, double s,
            bool eigenvectors){
    double** mat1;
    double** mat2;
    double** mat3;

    mat1 = (double**)malloc(__SIZEOF_POINTER__*2);
    mat1[0] = (double*)malloc(__SIZEOF_DOUBLE__*2);
    mat1[1] = (double*)malloc(__SIZEOF_DOUBLE__*2);
    mat1[0][0] = c; mat1[0][1] = -s;
    mat1[1][0] = s; mat1[1][1] = c;

    mat2 = (double**)malloc(__SIZEOF_POINTER__*2);
    mat2[0] = (double*)malloc(__SIZEOF_DOUBLE__*1);
    mat2[1] = (double*)malloc(__SIZEOF_DOUBLE__*1);
    if (eigenvectors){
        mat2[0][0] = E[i][k];
        mat2[1][0] = E[i][l];
    }
    else {
        mat2[0][0] = S[k][l];
        mat2[1][0] = S[i][j];
    }

    mat3 = mat_mul(mat1, 2, 2, mat2, 2, 1);

    if (eigenvectors){
        E[i][k] = mat3[0][0];
        E[i][l] = mat3[1][0];
    }
    else{
        S[k][l] = mat3[0][0];
        S[i][j] = mat3[1][0];
    }

    free(mat1[0]);
    free(mat1[1]);
    free(mat1);
    free(mat2[0]);
    free(mat2[1]);
    free(mat2);
    free(mat3[0]);
    free(mat3[1]);
    free(mat3);
}

void print_matrix(double** A, int Am, int An) {
    cout << "[";
    for (int i=0; i<Am; i++){
        if (i>0)
            cout<<" ";
        cout<<"[";
        for (int j=0; j<An-1; j++){
            cout << A[i][j] << ", ";
        }
        if (i < Am-1)
            cout << A[i][An-1] << "]" << endl;
    }
    cout << A[Am-1][An-1] << "]]" << endl;
}

void print_vector(double* A, int An) {
    cout << "[";
    for(int i=0; i<An-1; i++)
        cout << A[i] << ",";
    cout << A[An-1] << "]" << endl;
}

void init_jacobi() {
    E = (double**)malloc(__SIZEOF_POINTER__*N_jacobi);
    for (int i=0; i<N_jacobi; i++){
        E[i] = (double*)malloc(__SIZEOF_DOUBLE__*N_jacobi);
        for (int j=0; j<N_jacobi; j++){
            E[i][j] = 0;
        }
        E[i][i] = 1;
    }

    state = N_jacobi;

    e = (double*)malloc(__SIZEOF_DOUBLE__*N_jacobi);
    ind = (int*)malloc(__SIZEOF_INT__*N_jacobi);
    changed = (bool*)malloc(sizeof(bool)*N_jacobi);

    for (int k=0; k<N_jacobi; k++){
        ind[k]     = maxind(k);
        e[k]       = S[k][k];
        changed[k] = true;
    }
}

void Jacobi(double **input_matrix, int n, 
            double **eigenvalues, double ***eigenvectors) {
    N_jacobi = n;
    S = input_matrix;

    init_jacobi();

    while(state != 0){
        int m = 0;

        for (int k=1; k<N_jacobi-1; k++){
            if (fabs(S[k][ind[k]]) > fabs(S[m][ind[m]])){
                m = k;
            }
        }

        int k = m;
        int l = ind[m];
        double p = S[k][l];
        double y = (e[l] - e[k]) / 2.0;
        double d = fabs(y) + sqrt(p*p + y*y);
        double r = sqrt(p*p + d*d);
        double c = d / r;
        double s = p / r;
        double t = (p*p) / d;

        if (y < 0.0) { s = -s; t = -t; }

        S[k][l] = 0.0;
        update(k, -t);
        update(l, t);

        for (int i=0; i<k; i++)  { rotate(i, k, i, l, c, s, false); }
        for (int i=k+1; i<l; i++){ rotate(k, i, i, l, c, s, false); }
        for (int i=l+1; i<N_jacobi; i++)  { rotate(k, i, l, i, c, s, false); }

        for (int i=0; i<N_jacobi; i++){
            rotate(k, l, i, i, c, s, true);
        }

        ind[k] = maxind(k);
        ind[l] = maxind(l);
    }

    *eigenvalues = e;
    *eigenvectors = E;
}
void printMatrix(int m, int n, double ** mat){
    printf("printing matrix\n");
    for(int i=0;i<m;i++){
        for(int j=0;j<n;j++){
            printf("%.5f ",(*mat)[n*i+j]);
        }
        printf("\n");
    }
    printf("print over\n");
}
int multiply(int adim1, int adim2, double * a, int bdim1,int bdim2, double * b, double ** c){
    if(adim2!=bdim1){
        return -1;
    }
    for(int i1=0;i1<adim1;i1++){
        for(int j1=0;j1<bdim2;j1++){
            double temp=0.0;
            for(int i2=0;i2<adim2;i2++){
                temp += (a[adim2*i1+i2])*(b[bdim2*i2+ j1]);
            }
            (*c)[bdim2*i1+j1]=temp;
        }
    }
    return 0;
}

int customsort(int M, double * arr, int ** order){
    for(int i=0;i<M;i++){
        (*order)[i]=i;
    }
    int temp;
    for(int i=0;i<M-1;i++){
        for(int j=0;j<M-i-1;j++){
            if(arr[(*order)[j]]< arr[(*order)[j+1]]){
                temp = (*order)[j+1];
                (*order)[j+1]=(*order)[j];
                (*order)[j] = temp;
            }
        }
    }
    return 0;
}
void calctranspose(int M, int N, double* D, double** D_T){
    for(int i=0;i<N;i++){
        for(int j=0;j<M;j++){
            (*D_T)[M*i+j] = D[N*j+i];
        }
    }
}
void SVDori(int M, int N, double* D, double** U, double** SIGMA, double** V_T){
    *U = (double *)malloc(sizeof(double)* N*N);
    *SIGMA = (double *)malloc(sizeof(double) * N);
    *V_T = (double *)malloc(sizeof(double) * M*M);
    double **prod, *eigenvalues, **eigenvectors;
    double * D_T = (double *)malloc(sizeof(double) * N*M);
    calctranspose(M, N, D, &D_T);
    // printf("matrix is\n");
    // print_matrix(&D,M,N);
    double **D2d, **D_T2d;
    D2d = (double**)malloc(sizeof(double*)*M);
    for (int i=0; i<M; i++){
        D2d[i] = (double*)malloc(sizeof(double)*N);
    }
    for(int i=0;i<M;i++){
        for(int j=0;j<N;j++){
            D2d[i][j]=D[N*i+j];
        }
    }
    D_T2d = mat_transpose(D2d, M, N);
    prod = mat_mul(D2d, M, N, D_T2d, N, M);
    printf("prod is\n");
    print_matrix(prod,M,M);
    Jacobi(prod, M, &eigenvalues, &eigenvectors);
    cout << "\neigenvalues:" << endl;
    print_vector(eigenvalues, M);

    cout << "\neigenvectors:" << endl;
    print_matrix(eigenvectors, M,M);
    double * eigenvector = (double *)malloc(sizeof(double) * M*M);
    for(int i=0;i<M;i++){
        for(int j=0;j<M;j++){
            eigenvector[M*i+j]=eigenvectors[j][i];
        }
    }
    int * eigenvaluessortedorder = (int *)malloc(sizeof(int)*M);
    int statussorted = customsort(M,eigenvalues,&eigenvaluessortedorder);
    double * V = (double *)malloc(sizeof(double) * M*M);
    double * sigmamatrix = (double *)malloc(sizeof(double) * N*M);
    double * sigmainvmatrix = (double *)malloc(sizeof(double)*M*N);
    for(int i=0;i<M*N;i++){
        sigmainvmatrix[i]=0;
        sigmamatrix[i]=0;
    }
    for(int i=0;i<M*M;i++){
        V[i]=0;
        (*V_T)[i]=0;
    }
    for(int i=0;i<N;i++){
        double tempeigen = eigenvalues[eigenvaluessortedorder[i]];
        tempeigen = sqrt(tempeigen);
        (*SIGMA)[i] = (double) tempeigen;
        if(tempeigen==0){
            printf("division by zero eigen possible here ============================\n");
        }
        sigmamatrix[M*i+i] = tempeigen;
        sigmainvmatrix[N*i+i] = ((double) 1.0)/(tempeigen);
    }
    for(int i=0;i<M;i++){
        for(int j=0;j<M;j++){
            //i represents column, j represents row
            V[M*j+i] = eigenvectors[j][eigenvaluessortedorder[i]];
        }
    }
    calctranspose(M,M,V,V_T);
    double * tempmult = (double *)malloc(sizeof(double)*N*M);
    int statusmultiply = multiply(N,M,D_T,M,M,V,&tempmult);
    statusmultiply = multiply(N,M,tempmult,M,N,sigmainvmatrix,U);
    printf("U is\n");
    printMatrix(N,N,U);
    printf("V is\n");
    printMatrix(M,M,&V);
    printf("sigma is\n");
    printMatrix(N,M,&sigmamatrix);
    printf("M is\n");
    printMatrix(N,M,&D_T);
    printf("sigmainv is\n");
    printMatrix(M,N,&sigmainvmatrix);
    
}
void PCA(int retention, int M, int N, double* D, double* U, double* SIGMA, double** D_HAT, int *K)
{
    double sigmasum = 0.0;
    for(int i=0;i<N;i++){
        sigmasum+= (double) (SIGMA[i] * SIGMA[i]);
    }
    double targetthressigma = retention*sigmasum/100.0;
    double tempsigmasum = 0.0;
    int k=0;
    for(int i=0;i<N;i++){
        k+=1;
        tempsigmasum+= (double) (SIGMA[i] * SIGMA[i]);
        if(tempsigmasum>targetthressigma){
            break;
        }
    }
    *K=k;
    double * concatu = (double *)malloc(sizeof(double) * N*k);
    for(int i=0;i<N;i++){
        for(int j=0;j<k;j++){
            concatu[k*i+j] = (double) U[N*i+j];
        }
    }
    *D_HAT = (double *)malloc(sizeof(double) * M*k);
    int statusmultiply = multiply(M, N, D, N, k, concatu, D_HAT);
}


void SVD_and_PCA (int M, 
        int N, 
        double* D, 
        double** U, 
        double** SIGMA, 
        double** V_T, 
        int* SIGMAm,
        int* SIGMAn, 
        double** D_HAT, 
        int *K,
        int retention) {
    // write your code here
    SVDori(M, N, D, U, SIGMA, V_T);
    PCA(retention, M, N, D, *U, *SIGMA, D_HAT, K);
}

