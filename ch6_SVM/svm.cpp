// gcc version 11.2.0 (Rev10, Built by MSYS2 project)
// g++ -o svm.exe svm.cpp -D LOCAL -O3
#include <bits/stdc++.h>
using namespace std;
const int trainN = 574;
const double eps = 1e-6;
mt19937 gen(time(NULL));
vector<double> X[trainN];
double xtx[trainN][trainN];
double dis[trainN][trainN];
int Y[trainN];
double c, sumalpha, aayyxx;
double w[8];
double B;
vector<double> alpha(trainN, 0);

void updb()
{
    double tmp = 0.0;
    int cnt = 0;
    for (int k = 0; k < trainN; k++)
    {
        if(alpha[k]<eps)    continue;
        double tmpp = 0.0;
        cnt++;
        for (int i = 0; i < trainN; i++)
        {
            if (alpha[i] > eps)
                tmpp += alpha[i] * Y[i] * xtx[i][k];
        }
        tmp += (Y[k] - tmpp);
    }
    if (cnt == 0)
        return;
    B = tmp / cnt;
}

int main()
{
    // read train set.
    // ifstream alphain("alpha.out");
    // for(int i=0;i<trainN;i++)
    // {
    //     alphain>>alpha[i];
    // }
    ifstream trainfile("diabetes_train.data");
    for (int i = 0; i < trainN; i++)
    {
        double tmp;
        for (int j = 0; j < 8; j++)
        {
            trainfile >> tmp;
            trainfile.get();
            X[i].push_back(tmp);
        }
        string str;
        trainfile >> str;
        if (str == "tested_positive")
            Y[i] = 1;
        else
            Y[i] = -1;
    }
    // init
    for (int i = 0; i < trainN; i++)
    {
        c -= alpha[i] * Y[i];
        sumalpha += alpha[i];
    }
    auto polykernal = [](vector<double> a, vector<double> b,double n=1) -> double
    {
        double res = 0.0;
        for (int i = 0; i < 8; i++)
            res += a[i] * b[i];
        // cout<<res<<endl;
        return pow(res,n);
    };
    auto guasskernal = [](vector<double> a, vector<double> b) -> double
    {
        double res = 0.0;
        double sigma2 = 10;
        for (int i = 0; i < 8; i++)
            res += (a[i] - b[i]) * (a[i] - b[i]);
        res /=  sigma2;
        return exp(-1 * res);
    };
    auto distance = [](vector<double> a, vector<double> b) -> double
    {
        double res = 0.0;
        for (int i = 0; i < 8; i++)
            res += (a[i] - b[i]) * (a[i] - b[i]);
        // cout<<res<<endl;
        return res;
    };
    for (int i = 0; i < trainN; i++)
    {
        for (int j = 0; j <= i; j++)
        {
            xtx[i][j] = polykernal(X[i], X[j]);
            xtx[j][i] = xtx[i][j];
            dis[i][j] = distance(X[i],X[j]);
            dis[j][i] = dis[i][j];
        }
    }
    // object of dual
    double L = 0;
    // SMO
    // choose diff alpha T times
    int T = 100000;
    while (T--)
    {
        int a = gen() % trainN;
        double maxx=0;
        int b=0;
        for(int i=0;i<trainN;i++)
        {
            if(maxx<dis[i][a])
            {
                maxx=dis[i][a];
                b=i;
            }
        }
        if (a == b)
            continue;

        // check KKT condition(skip)
        updb();
        double fa = B;
        for (int i = 0; i < trainN; i++)
        {
            fa += Y[i] * alpha[i] * xtx[i][a];
        }
        if(Y[a]*fa>1.0&& alpha[a]==0) continue;
        // fa *= Y[a];

        double da = 0.0, db = 0.0, c = 0.0;
        for (int i = 0; i < trainN; i++)
        {
            if (i != a && i != b)
            {
                da += (xtx[a][i] * Y[i] * alpha[i]);
                db += (xtx[b][i] * Y[i] * alpha[i]);
                c -= (Y[i] * alpha[i]);
            }
        }
        const double C = 1;
        double lin_c = Y[b] * (Y[b] - Y[a] + C * (xtx[a][a] - xtx[a][b]) + da - db);
        double qd_c = xtx[a][a] + xtx[b][b] - 2.0 * xtx[a][b];
        // // upd method 1
        // // cout<<lin_c<<' '<<qd_c<<endl;
        // double alpha_b = lin_c / qd_c;
        // bool f=0;
        // if (alpha_b < 0)
        //     alpha_b = 0,f=1;
        // if (alpha_b > C)
        //     alpha_b = C,f=1;

        // double alpha_a = (c - alpha_b * Y[b]) * Y[a];

        // if (alpha_a < 0)
        //     alpha_a = 0,f=1;
        // if (alpha_a > C)
        //     alpha_a = C,f=1;

        // if(f)   continue;
        // alpha[a] = alpha_a, alpha[b] = alpha_b;

        // upd method 2
        double u, v, ea = B - Y[a], eb = B - Y[b];
        for (int i = 0; i < trainN; i++)
        {
            ea += xtx[i][a] * alpha[i] * Y[i];
            eb += xtx[i][b] * alpha[i] * Y[i];
        }
        double alpha_b = alpha[b] + Y[b] * (ea - eb) / qd_c;
        if (Y[a] * Y[b] == 1)
        {
            u = max(0.0, alpha[a] + alpha[b] - C);
            v = min(C, alpha[a] + alpha[b]);
        }
        else
        {
            u = max(0.0, alpha[b] - alpha[a]);
            v = min(C, C - alpha[a] + alpha[b]);
        }
        if(alpha_b>v||alpha_b<u)
        {
            continue;
        }
        alpha_b = min(alpha_b, v);
        alpha_b = max(alpha_b, u);
        double alpha_a = alpha[a] + Y[a] * Y[b] * (alpha[b] - alpha_b);
        alpha[a] = alpha_a, alpha[b] = alpha_b;

        if (T % 100 == 0)
        {
            double pre_L=L;
            L=0.0;
            for (int i = 0; i < trainN; i++)
            {
                L += alpha[i];
            }
            for (int i = 0; i < trainN; i++)
            {
                for (int j = 0; j < trainN; j++)
                {
                    L -= 0.5 * alpha[i] * alpha[j] * Y[i] * Y[j] * xtx[i][j];
                }
            }
            if(fabs(pre_L-L)<eps) break;
            cout << "L: " << L << endl;
        }
    }
    updb();
    double ans = 0.0;
    for (int i = 0; i < trainN; i++)
        ans += Y[i] * alpha[i];
    cout << ans << endl;
    ofstream outfile("alpha.out");
    int cnt = 0;
    for (int i = 0; i < trainN; i++)
    {
        outfile << alpha[i] << endl;
        if (alpha[i] > 1e-5)
            cnt++;
    }
    cout << "num of SV: " << cnt << endl;

    for (int i = 0; i < trainN; i++)
        for (int j = 0; j < 8; j++)
            w[j] += alpha[i] * Y[i] * X[i][j];

    cout << "---ans---" << endl;
    cout << "w:" << endl;
    for (int j = 0; j < 8; j++)
        cout << w[j] << endl;
    cout << "b:" << endl;
    cout << B << endl;

    // train data set


    int corr = 0;
    for (int i = 0; i < trainN; i++)
    {
        double res = 0.0;
        for (int j = 0; j < 8; j++)
        {
            res += X[i][j] * w[j];
        }
        res += B;
        if (res < 0 && Y[i] == -1 || res > 0 && Y[i] == 1)
            corr++;
    }
    cout << "corr: " << corr << endl;
    cout << "precision: " << (double)corr / trainN << endl;

    cout << "-----------\n";
    cout << "test set\n";
    int testcorr = 0;
    ifstream testfile("diabetes_test.data");
    const int testN = 194;
    for (int i = 0; i < testN; i++)
    {
        double tmp, res = 0.0;
        for (int j = 0; j < 8; j++)
        {
            testfile >> tmp;
            testfile.get();
            res += tmp * w[j];
        }
        res += B;
        string str;

        testfile >> str;
        if (str == "tested_positive" && res > 0)
            testcorr++;
        if (str == "tested_negative" && res < 0)
            testcorr++;
    }
    cout << "testcorr: " << testcorr << "\nprecison: ";
    cout << (double)testcorr / testN << endl;
}

/*
local:

num of SV: 505
---ans---
w:
0.0267714
0.0104161
-0.00604418
-0.00113693
0.00164941
0.0373179
0.00578085
0.0109524
b:
-2.98471
corr: 438
precision: 0.763066
-----------
test set
testcorr: 150
precison: 0.773196
*/

/*
server: T=10*local

num of SV: 503
---ans---
w:
0.0391774
0.00958592
-0.00496118
0.000720665
-0.00107507
0.0429765
0.0537018
0.0113769
b:
-3.01234
corr: 435
precision: 0.75784
-----------
test set
testcorr: 149
precison: 0.768041
*/