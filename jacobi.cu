#include "jacobi.h"

#define TOLERANCE 0.001
#define JACOBI_UPDATE_TOLERANCE 0.001
// int samples;
// int features;
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
    B = (double**)malloc(sizeof(double *)*An);
    for (int i=0; i<An; i++)
        B[i] = (double*)malloc(sizeof(double)*Am);

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
    C = (double**)malloc(sizeof(double *)*Am);
    for (int i=0; i<Am; i++)
        C[i] = (double*)malloc(sizeof(double)*Bn);

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
    // double** mat1;
    // double** mat2;
    // double** mat3;

    // mat1 = (double**)malloc(sizeof(double *)*2);
    // mat1[0] = (double*)malloc(sizeof(double)*2);
    // mat1[1] = (double*)malloc(sizeof(double)*2);
    // mat1[0][0] = c; mat1[0][1] = -s;
    // mat1[1][0] = s; mat1[1][1] = c;

    // mat2 = (double**)malloc(sizeof(double *)*2);
    // mat2[0] = (double*)malloc(sizeof(double)*1);
    // mat2[1] = (double*)malloc(sizeof(double)*1);
    // if (eigenvectors){
    //     mat2[0][0] = E[i][k];
    //     mat2[1][0] = E[i][l];
    // }
    // else {
    //     mat2[0][0] = S[k][l];
    //     mat2[1][0] = S[i][j];
    // }

    // mat3 = mat_mul(mat1, 2, 2, mat2, 2, 1);

    // if (eigenvectors){
    //     E[i][k] = mat3[0][0];
    //     E[i][l] = mat3[1][0];
    // }
    // else{
    //     S[k][l] = mat3[0][0];
    //     S[i][j] = mat3[1][0];
    // }

    // free(mat1[0]);
    // free(mat1[1]);
    // free(mat1);
    // free(mat2[0]);
    // free(mat2[1]);
    // free(mat2);
    // free(mat3[0]);
    // free(mat3[1]);
    // free(mat3);
    double mat0=0;
    double mat1=0;
    if(eigenvectors){
        mat0 = c * E[i][k] - s * E[i][l];
        mat1 = s * E[i][k] + c * E[i][l];
        E[i][k] = mat0;
        E[i][l] = mat1;
    }else{
        mat0 = c * S[k][l] - s * S[i][j];
        mat1 = s * S[k][l] + c * S[i][j];
        S[k][l] = mat0;
        S[i][j] = mat1;
    }
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
    E = (double**)malloc(sizeof(double *)*N_jacobi);
    for (int i=0; i<N_jacobi; i++){
        E[i] = (double*)malloc(sizeof(double)*N_jacobi);
        for (int j=0; j<N_jacobi; j++){
            E[i][j] = 0;
        }
        E[i][i] = 1;
    }

    state = N_jacobi;

    e = (double*)malloc(sizeof(double)*N_jacobi);
    ind = (int*)malloc(sizeof(int)*N_jacobi);
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

