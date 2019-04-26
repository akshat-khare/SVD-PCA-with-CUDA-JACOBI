#include <malloc.h>
#include <omp.h>
#include <math.h>
#include <stdio.h>
#include "jacobi.h"
#include "lab3_cuda.h"


// #include <stdio.h>
// #include <stdlib.h>
// #include <assert.h>

// #define BLOCK_SIZE 16
#define USEGPU 1


int debug =1;
int debug2 = 0;
int labid=3;
// int NUMTHREADS= 1;
int maxloops= 100000;
double convergencemetric = 0.0001;

void calctranspose(int M, int N, double* D, double** D_T){
    for(int i=0;i<N;i++){
        for(int j=0;j<M;j++){
            (*D_T)[M*i+j] = D[N*j+i];
        }
    }
}
void calctransposedouble(int M, int N, double* D, double** D_T){
    for(int i=0;i<N;i++){
        for(int j=0;j<M;j++){
            (*D_T)[M*i+j] = D[N*j+i];
        }
    }
}
double absfunc(double a, double b){
    if(a>b){
        return a-b;
    }else{
        return b-a;
    }
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
void printMatrixdouble(int m, int n, double ** mat){
    printf("printing matrix\n");
    for(int i=0;i<m;i++){
        for(int j=0;j<n;j++){
            printf("%.5f ",(*mat)[n*i+j]);
        }
        printf("\n");
    }
    printf("print over\n");
}
void printMatrixint(int m, int n, int ** mat){
    printf("printing matrix\n");
    for(int i=0;i<m;i++){
        for(int j=0;j<n;j++){
            printf("%d ",(*mat)[n*i+j]);
        }
        printf("\n");
    }
    printf("print over\n");
}


#include <stdio.h>
#include <stdlib.h>
#include <assert.h>

#define BLOCK_SIZE 16

__global__ void gpu_matrix_mult(double *a,double *b, double *c, int m, int n, int k)
{ 
    int row = blockIdx.y * blockDim.y + threadIdx.y; 
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    double sum = 0;
    if( col < k && row < m) 
    {
        for(int i = 0; i < n; i++) 
        {
            sum += a[row * n + i] * b[i * k + col];
        }
        c[row * k + col] = sum;
    }
} 
int multiplycpu(int adim1, int adim2, double * a, int bdim1,int bdim2, double * b, double ** c){
    
        if(adim2!=bdim1){
            return -1;
        }
        // #pragma omp parallel for collapse(2)

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

int multiply(int adim1, int adim2, double * a, int bdim1,int bdim2, double * b, double ** c){
    if(USEGPU==0){
        if(adim2!=bdim1){
            return -1;
        }
        // #pragma omp parallel for collapse(2)

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
    }else{
        if(adim2!=bdim1){
            return -1;
        }
        int m = adim1;
        int n = adim2;
        int k = bdim2;
        // printf("matrix mul 1\n");
        // printMatrix(m,n,&a);
        // printMatrix(n,k,&b);
        // printMatrix(m,k,c);

        double *d_a, *d_b, *d_c;
        // double * cbackup = (double *)malloc(sizeof(double)* m*k);
     //    cudaMallocHost((void **) &a, sizeof(double)*m*n);
    	// cudaMallocHost((void **) &b, sizeof(double)*n*k);
     //    cudaMallocHost((void **) c, sizeof(double)*m*k);
        // cudaMallocHost((void **) &h_cc, sizeof(int)*m*k);
        cudaMalloc((void **) &d_a, sizeof(double)*m*n);
        cudaMalloc((void **) &d_b, sizeof(double)*n*k);
        cudaMalloc((void **) &d_c, sizeof(double)*m*k);
        cudaMemcpy(d_a, a, sizeof(double)*m*n, cudaMemcpyHostToDevice);
        cudaMemcpy(d_b, b, sizeof(double)*n*k, cudaMemcpyHostToDevice);
        dim3 dimGrid((k + BLOCK_SIZE - 1) / BLOCK_SIZE, (m + BLOCK_SIZE - 1) / BLOCK_SIZE);
        dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
        gpu_matrix_mult<<<dimGrid, dimBlock>>>(d_a, d_b, d_c, m, n, k); 
        cudaMemcpy((*c), d_c, sizeof(double)*m*k, cudaMemcpyDeviceToHost);
        cudaThreadSynchronize();
        cudaFree(d_a);
        cudaFree(d_b);
        cudaFree(d_c);
        // for(int i=0;i<m*k;i++){
        // 	(*c)[i]=cbackup[i];
        // }
        // free(cbackup);
        // cudaFreeHost(a);
        // cudaFreeHost(b);
        // cudaFreeHost(*c);
        // cudaFreeHost(h_cc);
        // printf("matrix mul 2\n");
        // printMatrix(m,n,&a);
        // printMatrix(n,k,&b);
        // printMatrix(m,k,c);
        return 0;
    }

}
int subtract(int adim1, int adim2, double * a, int bdim1,int bdim2, double * b, double ** c){
    if(adim1!=bdim1 || adim2!=bdim2){
        return -1;
    }
    for(int i=0;i<adim1;i++){
        for(int j=0;j<adim2;j++){
            (*c)[adim2*i+j] = a[adim2*i+j]-b[adim2*i+j];
        }
    }
    return 0;
}
int subtractdiag(int adim1, int adim2, double * a, int bdim1,int bdim2, double * b, double ** c){
    if(adim1!=bdim1 || adim2!=bdim2){
        return -1;
    }
    for(int i=0;i<adim1;i++){
        // for(int j=0;j<adim2;j++){
        (*c)[i] = a[adim2*i+i]-b[adim2*i+i];
        // }
    }
    return 0;
}

double sumsquareelements(int M, int N, double *m){
    double temp=0.0;
    for(int i=0;i<M;i++){
        for(int j=0;j<N;j++){
            temp += (m[N*i+j])*(m[N*i+j]);
        }
    }
    return temp;
}

double sumabsoelements(int M, int N, double *m ){
    double temp=0.0;
    for(int i=0;i<M;i++){
        for(int j=0;j<N;j++){
            temp += absfunc(m[N*i+j],0.0);
        }
    }
    return temp;
}

double maxabsoelements(int M, int N, double *m ){
    double temp= absfunc(m[0],0.0);
    for(int i=0;i<M;i++){
        for(int j=0;j<N;j++){
            if(absfunc(m[N*i+j],0.0)>temp){
                temp = absfunc(m[N*i+j],0.0);
            }
        }
    }
    return temp;
}

double maxdiagabsoelements(int M, int N, double *m ){
    double temp= absfunc(m[0],0.0);
    for(int i=0;i<M;i++){
        // for(int j=0;j<N;j++){
        if(absfunc(m[N*i+i],0.0)>temp){
            temp = absfunc(m[N*i+i],0.0);
        }
        // }
    }
    return temp;
}
int maxdiagabsoelementscmp(int M, int N, double *m ,double convf){
    // double temp= convf;
    int status = 1;
    for(int i=0;i<M;i++){
        // for(int j=0;j<N;j++){
        if(absfunc(m[i],0.0)>convf){
            // temp = absfunc(m[N*i+i],0.0);
            status=-1;
            break;
        }
        // }
    }
    return status;
}


double proj(int M,double ** a, int j, double ** e, int k){
    double temp =0.0;
    for(int i=0;i<M;i++){
        temp += ((*a)[M*i+j])*((*e)[M*i+k]);
    }
    return temp;
}
double norm(int M, double ** u, int j){
    double temp=0.0;
    for(int i=0;i<M;i++){
        temp += ((*u)[M*i+j])*((*u)[M*i+j]);
    }
    temp = sqrt(temp);
    return temp;

}
int qrfactors(int M, double * a, double ** q, double ** r){
    double * u = (double *)malloc(sizeof(double) * M *M);
    double * e = (double *)malloc(sizeof(double) * M *M);
    for(int j=0;j<M;j++){
        for(int i=0;i<M;i++){
            u[M*i+j] = a[M*i+j];
        }
        for(int diffell=0; diffell<j;diffell++){
            double tempproj = proj(M,&a,j,&e,diffell);
            // (*r)[M*diffell+j]=tempproj;
            for(int i=0;i<M;i++){
                u[M*i+j] = u[M*i+j] - tempproj*(e[M*i+diffell]);
            }
        }
        double normuj= norm(M,&u,j);
        if(normuj==0){
            if(0==debug) printf("division by zero possible here\n");
            for(int i=0;i<M;i++){
                e[M*i+j]=0;
            }
        }else{
            for(int i=0;i<M;i++){
                e[M*i+j]=(1.0/normuj)*(u[M*i+j]);
            }
        }
        
        // (*r)[M*j+j]=proj(M,&a,j,&e,j);
        // (*r)[M*j+j]=normuj;

    }
    for(int i=0;i<M;i++){
        for(int j=0;j<M;j++){
            (*q)[M*i+j] = e[M*i+j];
            // if(i>j){
            //     (*r)[M*i+j]=0;
            // }
        }
    }
    if(0==debug){

        double * q_tm = (double *)malloc((sizeof(double) *M*M));
        calctranspose(M,M,*q,&q_tm);
        int statusmultiply = multiply(M,M,q_tm,M,M,a,r);
        printf("q is\n");
        printMatrix(M,M,q);
        printf("r is\n");
        printMatrix(M,M,r);
        double * qmulr = (double *)malloc(sizeof(double) * M *M);
        multiply(M,M,*q,M,M,*r,&qmulr);
        printf("q x r is\n");
        printMatrix(M,M,&qmulr);
        printf("original matrix is\n");
        printMatrix(M,M,&a);
        free(q_tm);
        free(qmulr);
    }
    free(u);
    free(e);
    return 0;

}


int qrmodifiedfactors(int M, double * a, double ** q, double ** r){
    double * v = (double *)malloc(sizeof(double) * M *M);
    // double * e = (double *)malloc(sizeof(double) * M *M);
    for(int i=0;i<M;i++){
        for(int j=0;j<M;j++){
            v[M*i+j]=a[M*i+j];
            (*r)[M*i+j]=0.0;
        }
    }
    for(int i=0;i<M;i++){
        double tempnorm = norm(M,&v,i);
        if(tempnorm==0){
            if(0==debug) printf("division by zero being done\n");
        }
        (*r)[M*i+i]= tempnorm;
        for(int rowiter=0;rowiter<M;rowiter++){
            (*q)[M*rowiter+i] = (1.0/tempnorm)*(v[M*rowiter+i]);
        }
        // #pragma omp parallel for
        for(int j=i+1;j<M;j++){
            double rij = proj(M,q,i,&v,j);
            (*r)[M*i+j] = rij;
            for(int rowiter=0;rowiter<M;rowiter++){
                v[M*rowiter+j] = v[M*rowiter+j] - rij*((*q)[M*rowiter+i]);
            }

        }

    }
    free(v);
    // double * q_tm = (double *)malloc((sizeof(double) *M*M));
    // calctranspose(M,M,*q,&q_tm);
    // int statusmultiply = multiply(M,M,q_tm,M,M,a,r);
    if(0==debug){

        // printf("q is\n");
        // printMatrix(M,M,q);
        // printf("r is\n");
        // printMatrix(M,M,r);
        double * qmulr = (double *)malloc(sizeof(double) * M *M);
        multiply(M,M,*q,M,M,*r,&qmulr);
        // printf("q x r is\n");
        // printMatrix(M,M,&qmulr);
        // printf("original matrix is\n");
        // printMatrix(M,M,&a);
        // free(q_tm);
        double * diffm = (double *)malloc(sizeof(double) * M *M);
        subtract(M,M,a,M,M,qmulr,&diffm);
        double tempabsodiff = sumabsoelements(M,M,diffm);
        printf("Absolute diff qr is %.6f -------------------------------\n",tempabsodiff);
        if(tempabsodiff>0.001){
            printf("q is\n");
            printMatrix(M,M,q);
            printf("r is\n");
            printMatrix(M,M,r);
            printf("q x r is\n");
            printMatrix(M,M,&qmulr);
            printf("original matrix is\n");
            printMatrix(M,M,&a);
            printf("diff matrix is \n");
            printMatrix(M,M,&diffm);
        }
        free(diffm);
        free(qmulr);
    }
    return 0;

}


int findeigen(int M, double * darg, double ** eigenvector, double ** eigenvalues){
    if(debug==0) printf("original d is \n");
    if(debug==0) printMatrix(M,M,&darg);
    double * d_eval = (double *)malloc(sizeof(double) * M*M);
    for(int i=0;i<M;i++){
        for(int j=0;j<M;j++){
            d_eval[M*i+j]=darg[M*i+j];
        }
    }
    double * e_evec = (double *)malloc(sizeof(double) * M*M);
    for(int i=0;i<M;i++){
        for(int j=0;j<M;j++){
            if(i==j){
                e_evec[M*i+j]=1;
            }else{
                e_evec[M*i+j]=0;
            }
        }
    }
    double * d_evalnew = (double *)malloc(sizeof(double) * M*M);
    for(int i=0;i<M;i++){
        for(int j=0;j<M;j++){
            d_evalnew[M*i+j]=darg[M*i+j];
        }
    }
    double * e_evecnew = (double *)malloc(sizeof(double) * M*M);
    for(int i=0;i<M;i++){
        for(int j=0;j<M;j++){
            if(i==j){
                e_evecnew[M*i+j]=1;
            }else{
                e_evecnew[M*i+j]=0;
            }
        }
    }
    int numloop=0;
    double * qmat = (double *)malloc(sizeof(double) * M*M);
    double * rmat = (double *)malloc(sizeof(double) * M*M);
    int numchangesd=0;
    int numchangese=0;
    int statusqr;
    int statusmultiply;
    double * ddiff = (double *)malloc(sizeof(double) * M);
    // double * ediff = (double *)malloc(sizeof(double) * M*M);
    while(0==0){
        if(0==debug) printf("loop %d starting\n",numloop);
        statusqr = qrmodifiedfactors(M, d_eval, &qmat, &rmat);


        statusmultiply = multiply(M,M,rmat,M,M,qmat, &d_evalnew);
        statusmultiply = multiply(M,M,e_evec,M,M,qmat,&e_evecnew);


        if(debug==0) printf("D_eval is \n");
        if(debug==0) printMatrix(M,M,&d_eval);
        if(debug==0) printf("D_evalnew is \n");
        if(debug==0) printMatrix(M,M,&d_evalnew);
        if(debug==0) printf("e_eval is \n");
        if(debug==0) printMatrix(M,M,&e_evec);
        if(debug==0) printf("e_evalnew is \n");
        if(debug==0) printMatrix(M,M,&e_evecnew);
        // if(debug==0) printf("q is \n");
        // if(debug==0) printMatrix(M,M,&qmat);
        // if(debug==0) printf("r is \n");
        // if(debug==0) printMatrix(M,M,&rmat);

        // numchangesd=0;
        // numchangese=0;
        // double tempdiff1=0.0;
        // double tempdiff2=0.0;
        // for(int i=0;i<M;i++){
        //     for(int j=0;j<M;j++){
        //         // if(absfunc(d_evalnew[M*i+j],d_eval[M*i+j])>0.0001){
        //         //     numchangesd+=1;
        //         // }
        //         // if(absfunc(e_evecnew[M*i+j],e_evec[M*i+j])>0.0001){
        //         //     numchangese+=1;
        //         // }
        //         tempdiff1 += absfunc(d_evalnew[M*i+j],d_eval[M*i+j]);
        //         tempdiff2 += absfunc(e_evecnew[M*i+j],e_evec[M*i+j]);
        //     }
        // } 

        subtractdiag(M,M,d_evalnew,M,M,d_eval,&ddiff);
        if(0==debug) printMatrix(1,M,&ddiff);
        // subtract(M,M,e_evecnew,M,M,e_evec,&ediff);
        int maxddiffstatus = maxdiagabsoelementscmp(M,M,ddiff,convergencemetric);
        // double maxediff = maxabsoelements(M,M,ediff);

        // #pragma omp parallel for collapse(2)
        for(int i=0;i<M;i++){
            for(int j=0;j<M;j++){
                d_eval[M*i+j]=d_evalnew[M*i+j];
                e_evec[M*i+j]=e_evecnew[M*i+j];
            }
        } 
        numloop+=1;
        // if(0==debug) printf("loop %d ending with numchangesd %d and numchangese %d\n",numloop, numchangesd, numchangese);
        if(0==debug) printf("loop %d ending with maxddiff %d\n",numloop, maxddiffstatus);        
        if(0==debug2) printf("loop %d ending with maxddiff %d\n",numloop, maxddiffstatus);        
        
        // if(0==debug2) printf("eigen %d loop with diff %.6f %.6f\n",numloop, tempdiff1, tempdiff2);
        // if(tempdiff1 < 0.001 && tempdiff2<0.001){
        //     if(0==debug) printf("breaking on loop %d\n",numloop);
        //     if(0==debug2) printf("breaking on loop %d\n",numloop);
        //     break;
        // }
        if(maxddiffstatus==1){
            if(0==debug) printf("breaking on loop %d\n",numloop);
            if(0==debug2) printf("breaking on loop %d\n",numloop);
            break;
        }
        if(numloop>maxloops){
            // if(0==debug) printf("eigen end loop with diff %.6f %.6f\n", tempdiff1, tempdiff2);
            // if(0==debug2) printf("eigen end loop with diff %.6f %.6f\n", tempdiff1, tempdiff2);
            if(0==debug) printf("eigen end loop with diff %d\n", maxddiffstatus);
            if(0==debug2) printf("eigen end loop with diff %d\n", maxddiffstatus);
            
            break;
        }
    }
    if(0==debug) printf("D after convergence is\n");
    if(0==debug) printMatrix(M,M,&d_eval);
    if(0==debug) printf("E after convergence is \n");
    if(0==debug) printMatrix(M,M,&e_evec);
    for(int i=0;i<M;i++){
        (*eigenvalues)[i]=d_eval[M*i+i];
        for(int j=0;j<M;j++){
            (*eigenvector)[M*i+j]=e_evec[M*i+j];
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

void SVDreal(int M, int N, double* D, double** U, double** SIGMA, double** V_T)
{
    // omp_set_num_threads(NUMTHREADS);
    //Sigma is not in matrix form
    //First calculate d_t
    double * D_T = (double *)malloc(sizeof(double) * N*M);
    double * Ddouble = (double *)malloc(sizeof(double) * M*N);
    for(int i=0;i<M*N;i++){
        Ddouble[i] =(double) D[i];
    }
    calctranspose(M, N, Ddouble, &D_T);
    //now we need to calculate svd of d_t
    //m of example is d_t
    //m_t of example is D

    //need to find m_t.m which is d.d_t
    double * d_multiply_d_t = (double *)malloc(sizeof(double) * M*M);
    int statusmultiply = multiply(M, N, Ddouble, N, M, D_T, &d_multiply_d_t);

    //need to find the eigen values of d_multiply_d_t
    double * eigenvector = (double *)malloc(sizeof(double) * M*M);
    double * eigenvalues = (double *)malloc((sizeof(double) * M));
    // int statuseigen = findeigen(M, d_multiply_d_t, &eigenvector, &eigenvalues);
    if(labid==2){
        int statuseigen = findeigen(M, d_multiply_d_t, &eigenvector, &eigenvalues);
    }else if(labid==3){

        double ** d_multiply_d_tmat = (double**)malloc(sizeof(double *)*M);
        for(int i=0;i<M;i++){
            d_multiply_d_tmat[i] = (double *)malloc(sizeof(double)*M);
            for(int j=0;j<M;j++){
                d_multiply_d_tmat[i][j] = d_multiply_d_t[M*i+j];
            }
        }
        double ** eigenvectormat = (double **)malloc(sizeof(double)*M);
        for (int i=0;i<M;i++){
            eigenvectormat[i] = (double *)malloc(sizeof(double)*M);
        }
        Jacobi(d_multiply_d_tmat, M,&eigenvalues,&eigenvectormat);
        for(int i=0;i<M;i++){
            for(int j=0;j<M;j++){
                eigenvector[M*i+j] = eigenvectormat[i][j];
            }
        }
    }
    int * eigenvaluessortedorder = (int *)malloc(sizeof(int)*M);
    for(int i=0;i<M;i++){
        eigenvaluessortedorder[i]=i;
    }
    
    int statussorted = customsort(M,eigenvalues,&eigenvaluessortedorder);
    if(0==debug) printf("Sort order is\n");
    if(0==debug) printMatrixint(1,M,&eigenvaluessortedorder);
    double * sigmamatrix = (double *)malloc(sizeof(double) * N*M);
    double * sigmainvmatrix = (double *)malloc(sizeof(double)*M*N);
    double * V = (double *)malloc(sizeof(double) * M *M);
    double * V_Tdouble = (double *)malloc(sizeof(double) * M *M);
    double * Udouble = (double *)malloc(sizeof(double) * N *N);
    for(int i=0;i<M*N;i++){
        sigmainvmatrix[i]=0;
        sigmamatrix[i]=0;
    }
    for(int i=0;i<M*M;i++){
        V[i]=0;
        V_Tdouble[i]=0;
    }
    // for(int i=0;i<N*N;i++){
    //     Udouble[i]=0;
    // }
    for(int i=0;i<M;i++){
        double tempeigen = eigenvalues[eigenvaluessortedorder[i]];
        tempeigen = sqrt(tempeigen);
        (*SIGMA)[i] = (double) tempeigen;
        if(tempeigen==0){
            if(0==debug) printf("division by zero eigen possible here ============================\n");
        }
        sigmamatrix[M*i+i] = tempeigen;
        // if()
        sigmainvmatrix[N*i+i] = ((double) 1.0)/(tempeigen);
        
    }
    for(int i=0;i<M;i++){
        for(int j=0;j<M;j++){
            V[M*j+i] = eigenvector[M*j+eigenvaluessortedorder[i]];
        }
    }
    calctranspose(M,M,V,&V_Tdouble);
    for(int i=0;i<M*M;i++){
        (*V_T)[i] = (double) (V_Tdouble[i]);
    }
    double * tempmult = (double *)malloc(sizeof(double)*N*M);
    statusmultiply = multiply(N,M,D_T,M,M,V,&tempmult);
    statusmultiply = multiply(N,M,tempmult,M,N,sigmainvmatrix,&Udouble);
    for(int i=0;i<N*N;i++){
        (*U)[i] = (double) (Udouble[i]);
    }
    if(0==debug){
        double * tempmult2 = (double *)malloc(sizeof(double)*N*M);
        statusmultiply = multiply(N,N,Udouble,N,M,sigmamatrix,&tempmult);
        statusmultiply = multiply(N,M,tempmult,M,M,V_Tdouble,&tempmult2);
        printf("U is\n");
        printMatrixdouble(N,N,U);
        printf("V_T is\n");
        printMatrixdouble(M,M,V_T);
        printf("Sigma matrix is\n");
        printMatrix(N,M,&sigmamatrix);
        printf("Sigma inv is\n");
        printMatrix(M,N,&sigmainvmatrix);
        printf("Sigma is\n");
        printMatrixdouble(1,N,SIGMA);
        printf("usigmavt is \n");
        printMatrix(N,M,&tempmult2);
        printf("ori m or d_t was");
        printMatrix(N,M,&D_T);
        printf("done svd\n");
    }
    if(0==debug2){
        double * tempmult3 = (double *)malloc(sizeof(double)*N*M);
        statusmultiply = multiply(N,N,Udouble,N,M,sigmamatrix,&tempmult);
        statusmultiply = multiply(N,M,tempmult,M,M,V_Tdouble,&tempmult3);
        double * tempmult4 = (double *)malloc(sizeof(double)*N*M);
        int statussubtract = subtract(N,M,D_T,N,M,tempmult3,&tempmult4);
        double sumsquare= sumsquareelements(N,M,tempmult4);
        printf("Subtract Matrix is\n");
        printMatrix(N,M,&tempmult4);
        double maxabsoele = maxabsoelements(N,M,tempmult4);
        printf("sumsquare is %.6f after divided is %.6f max diff is %.6f\n ", sumsquare, sumsquare/(N*M), maxabsoele);
        //tempmult3 is u*sigma*v_t
        
    }

}
// /*
// 	*****************************************************
// 		TODO -- You must implement this function
// 	*****************************************************
// */
void SVD(int M, int N, double* D, double** U, double** SIGMA, double** V_T)
{
    double * D_T = (double *)malloc(sizeof(double) * N*M);
    // double * Ddouble = (double *)malloc(sizeof(double) * M*N);
    // for(int i=0;i<M*N;i++){
    //     Ddouble[i] =(double) D[i];
    // }
    calctransposedouble(M, N, D, &D_T);
    double * Vinnerout = (double *)malloc(sizeof(double) * M*M);
    double * U_Tinnerout = (double *)malloc(sizeof(double) * N*N);
    double * sigmainner= (double *)malloc((sizeof(double)*M));
    SVDreal(N,M,D_T,&Vinnerout,&sigmainner,&U_Tinnerout);
    for(int i =0;i<N;i++){
        (*SIGMA)[i] = sigmainner[i];
    }
    for(int i=0;i<N;i++){
        for(int j=0;j<N;j++){
            (*U)[N*i+j]=U_Tinnerout[N*j+i];
        }
    }
    for(int i=0;i<M;i++){
        for(int j=0;j<M;j++){
            (*V_T)[M*i+j]=Vinnerout[M*j+i];
        }
    }
    if(0==debug2){
        double * Udouble = (double *)malloc(sizeof(double) * N *N);
        for(int i=0;i<N;i++){
            for(int j=0;j<N;j++){
                Udouble[N*i+j] = (double) ((*U)[N*i+j]);
            }
        }
        double * sigmamatrix = (double *)malloc(sizeof(double) * N*M);
        for(int i=0;i<N*M;i++){
            sigmamatrix[i]=0;
        }
        for(int i=0;i<N;i++){
            sigmamatrix[M*i+i]=(double) ((*SIGMA)[i]);
        }
        double * V_Tdouble = (double *)malloc(sizeof(double) * M *M);
        for(int i=0;i<M*M;i++){
            V_Tdouble[i]=(double) ((*V_T)[i]);
        }
        double * D_Tdouble = (double *)malloc(sizeof(double) * N*M);
        for(int i=0;i<N*M;i++){
            D_Tdouble[i] = (double) (D_T[i]);
        }
        double * tempmult = (double *)malloc(sizeof(double)*N*M);
        printf("now time for the real svd calc\n");
        double * tempmult3 = (double *)malloc(sizeof(double)*N*M);
        int statusmultiply = multiply(N,N,Udouble,N,M,sigmamatrix,&tempmult);
        statusmultiply = multiply(N,M,tempmult,M,M,V_Tdouble,&tempmult3);
        double * tempmult4 = (double *)malloc(sizeof(double)*N*M);
        int statussubtract = subtract(N,M,D_Tdouble,N,M,tempmult3,&tempmult4);
        double sumsquare= sumsquareelements(N,M,tempmult4);
        double maxabsoele = maxabsoelements(N,M,tempmult4);
        
        printf("Subtract Matrix is\n");
        printMatrix(N,M,&tempmult4);
        printf("sumsquare is %.6f after divided is %.6f max diff is %.6f\n ", sumsquare, sumsquare/(N*M), maxabsoele);
        //tempmult3 is u*sigma*v_t
        
    }


}

// /*
// 	*****************************************************
// 		TODO -- You must implement this function
// 	*****************************************************
// */
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
    double * D_HATdouble = (double *)malloc(sizeof(double) * M*k);
    double * Ddouble = (double *)malloc(sizeof(double) * M*N);
    for(int i=0;i<M*N;i++){
        Ddouble[i] = (double) D[i];
    }
    int statusmultiply = multiply(M, N, Ddouble, N, k, concatu, &D_HATdouble);
    for(int i=0;i<M*k;i++){
        (*D_HAT)[i] = (double) (D_HATdouble[i]);
    }
    if(0==debug){

        printf("D is\n");
        printMatrixdouble(M,N,&D);
        printf("D_Hat is\n");
        printMatrixdouble(M,k,D_HAT);
        printf("k is %d\n",k);
        printf("pca done\n");
    }
    if(0==debug2){
        printf("D_Hat is\n");
        printMatrixdouble(M,k,D_HAT);
        printf("k is %d\n",k);
        printf("hooola bhoola\n");
    }
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
    *U = (double*) malloc(sizeof(double) * N*N);
	*SIGMA = (double*) malloc(sizeof(double) * N);
	*V_T = (double*) malloc(sizeof(double) * M*M);
    SVD(M, N, D, U, SIGMA, V_T);
    PCA(retention, M, N, D, *U, *SIGMA, D_HAT, K);
}