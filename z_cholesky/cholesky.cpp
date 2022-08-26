//
// Created by liuxh on 22-7-11.
//
#include <iostream>
#include <string.h>
#include <stdio.h>
#include <vector>
#include <math.h>
#include "Eigen/Core"

using namespace std;
const int N = 1005;
typedef double Type;

Type A[N][N], L[N][N];

/** 分解A得到A = L * L^T */
void Cholesky(const Type A[][N], Type L[][N], int n)
{
    for(int k = 0; k < n; k++)
    {
        Type sum = 0;
        for(int i = 0; i < k; i++)
            sum += L[k][i] * L[k][i];
        sum = A[k][k] - sum;
        L[k][k] = sqrt(sum > 0 ? sum : 0);
        for(int i = k + 1; i < n; i++)
        {
            sum = 0;
            for(int j = 0; j < k; j++)
                sum += L[i][j] * L[k][j];
            L[i][k] = (A[i][k] - sum) / L[k][k];
        }
        for(int j = 0; j < k; j++)
            L[j][k] = 0;
    }
}

/** 回带过程 */
vector<Type> Solve(Type L[][N], vector<Type> X, int n)
{
    /** LY = B  => Y */
    for(int k = 0; k < n; k++)
    {
        for(int i = 0; i < k; i++)
            X[k] -= X[i] * L[k][i];
        X[k] /= L[k][k];
    }
    /** L^TX = Y => X */
    for(int k = n - 1; k >= 0; k--)
    {
        for(int i = k + 1; i < n; i++)
            X[k] -= X[i] * L[i][k];
        X[k] /= L[k][k];
    }
    return X;
}

void Print(Type L[][N], const vector<Type> B, int n)
{
    for(int i = 0; i < n; i++)
    {
        for(int j = 0; j < n; j++)
            cout<<L[i][j]<<" ";
        cout<<endl;
    }
    cout<<endl;
    vector<Type> X = Solve(L, B, n);
    vector<Type>::iterator it;
    for(it = X.begin(); it != X.end(); it++)
        cout<<*it<<" ";
    cout<<endl;
}

int main()
{
    int n;
    cin>>n;
    memset(L, 0, sizeof(L));
    for(int i = 0; i < n; i++)
    {
        for(int j = 0; j < n; j++)
            cin>>A[i][j];
    }
    vector<Type> B;
    for(int i = 0; i < n; i++)
    {
        Type y;
        cin>>y;
        B.push_back(y);
    }

//    n = 4;
//    Eigen::MatrixXd m(4, 4);
//    vector<Type> B;
//    Type A[N][N], L[N][N];
//
//    m << 4, -2, 4, 2,
//    -2, 10, -2, -7,
//    4, -2, 8, 4,
//    2, -7, 4, 7;
//    double* C = m.data();

    Cholesky(A, L, n);
    Print(L, B, n);
    return 0;
}

/**data**
4
4 -2 4 2
-2 10 -2 -7
4 -2 8 4
2 -7 4 7
8 2 16 6
*/