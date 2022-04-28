#include <boost/python.hpp>
#include <bits/stdc++.h>
using namespace std;
class svm
{
public:
    int trainN = 574;
    int wN = 8;
    double eps = 1e-6;

    // vector<double> X[trainN];
    vector<vector<double>> X = vector<vector<double>>(trainN, vector<double>(wN, 0.0));
    vector<vector<double>> xtx = vector<vector<double>>(trainN, vector<double>(trainN, 0.0));
    vector<vector<double>> dis = vector<vector<double>>(trainN, vector<double>(trainN, 0.0));
    // double xtx[trainN][trainN];
    // double dis[trainN][trainN];
    // int Y[trainN];
    vector<int> Y = vector<int>(trainN);
    double c=0.0, sumalpha=0.0, aayyxx=0.0;
    // double w[wN];
    vector<double> w = vector<double>(wN, 0.0);
    double B=0.0;
    vector<double> alpha = vector<double>(trainN, 0);
    string traindataname, testdataname;

    void updb();
    void readtrain(string);
    void train(int T);
    void test(string,int);
    svm(int,int,string);
};
svm::svm(int _trainN,int _wN,string trainpath):trainN(_trainN),wN(_wN),traindataname(trainpath){}
void svm::updb()
{
    double tmp = 0.0;
    int cnt = 0;
    for (int k = 0; k < trainN; k++)
    {
        if (alpha[k] < eps)
            continue;
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
void svm::readtrain(string str)
{
    // cout<<xtx.size()<<endl;
    ifstream trainfile(str);
    for (int i = 0; i < trainN; i++)
    {
        double tmp;
        for (int j = 0; j < wN; j++)
        {
            trainfile >> tmp;
            trainfile.get();
            // X[i].push_back(tmp);
            X[i][j]=tmp;
            // cout<<tmp<<' ';
        }
        string str;
        trainfile >> str;
        if (str == "1")
            Y[i] = 1;
        else
            Y[i] = -1;
    }
    cout<<"--finish reading---\n";
    cout<<"Xsize: "<<X.size()<<"x"<<X[0].size()<<endl;
    cout<<"Ysize: "<<Y.size()<<endl;
}
void svm::train(int T)
{
    mt19937 gen(time(NULL));
    readtrain(traindataname);
    // cout<<traindataname<<endl;
    // for(int i=0;i<wN;i++)   cout<<X[1][i]<<' ';
    // cout<<endl;
    for (int i = 0; i < trainN; i++)
    {
        c -= alpha[i] * Y[i];
        sumalpha += alpha[i];
    }
    auto polykernal = [this](vector<double> a, vector<double> b, double n = 1) -> double
    {
        double res = 0.0;
        for (int i = 0; i < wN; i++)
            res += a[i] * b[i];
        // cout<<res<<endl;
        return pow(res, n);
    };
    auto guasskernal = [this](vector<double> a, vector<double> b) -> double
    {
        double res = 0.0;
        double sigma2 = 10;
        for (int i = 0; i < wN; i++)
            res += (a[i] - b[i]) * (a[i] - b[i]);
        res /= sigma2;
        return exp(-1 * res);
    };
    auto distance = [this](vector<double> a, vector<double> b) -> double
    {
        double res = 0.0;
        for (int i = 0; i < wN; i++)
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
            dis[i][j] = distance(X[i], X[j]);
            dis[j][i] = dis[i][j];
            // cout<<"distance(X[i], X[j]): "<<distance(X[i], X[j])<<endl;
        }
    }
    // object of dual
    double L = 0;
    // SMO
    // choose diff alpha T times
    // int T = 10000;
    while (T--)
    {
        int a = gen() % (long unsigned int)trainN;
        double maxx = 0;
        int b = 0;
        for (int i = 0; i < trainN; i++)
        {
            if (maxx < dis[i][a])
            {
                maxx = dis[i][a];
                b = i;
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
        if (Y[a] * fa > 1.0 && alpha[a] == 0)
            continue;
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
        // double lin_c = Y[b] * (Y[b] - Y[a] + C * (xtx[a][a] - xtx[a][b]) + da - db);
        double qd_c = xtx[a][a] + xtx[b][b] - 2.0 * xtx[a][b];

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
        if (alpha_b > v || alpha_b < u)
        {
            continue;
        }
        alpha_b = min(alpha_b, v);
        alpha_b = max(alpha_b, u);
        double alpha_a = alpha[a] + Y[a] * Y[b] * (alpha[b] - alpha_b);
        alpha[a] = alpha_a, alpha[b] = alpha_b;

        if (T % 100 == 0)
        {
            double pre_L = L;
            L = 0.0;
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
            if (fabs(pre_L - L) < eps)
                break;
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
        for (int j = 0; j < wN; j++)
            w[j] += alpha[i] * Y[i] * X[i][j];

    cout << "---train---" << endl;
    // cout << "w:" << endl;
    // for (int j = 0; j < wN; j++)
    //     cout << w[j] << endl;
    // cout << "b:" << endl;
    // cout << B << endl;

    // train data set

    int corr = 0;
    for (int i = 0; i < trainN; i++)
    {
        double res = 0.0;
        for (int j = 0; j < wN; j++)
        {
            res += X[i][j] * w[j];
        }
        res += B;
        if (res < 0 && Y[i] == -1 || res > 0 && Y[i] == 1)
            corr++;
    }
    cout << "corr: " << corr << endl;
    cout << "precision: " << (double)corr / trainN << endl;
}
void svm::test(string testpath,int testnum)
{
    cout << "-----------\n";
    cout << "test set\n";
    int testcorr = 0;
    ifstream testfile(testpath);
    cout<<"testpath "<<testpath<<endl;
    const int testN = testnum;
    for (int i = 0; i < testN; i++)
    {
        double tmp, res = 0.0;
        for (int j = 0; j < wN; j++)
        {
            testfile >> tmp;
            testfile.get();
            res += tmp * w[j];
        }
        res += B;
        string str;

        testfile >> str;
        if (str == "1" && res > 0)
            testcorr++;
        if (str == "0" && res < 0)
            testcorr++;
        cout<<"str: "<<str<<"   ---   "<<res<<endl;
    }
    cout << "testcorr: " << testcorr << "\nprecison: ";
    cout << (double)testcorr / testN << endl;
}

BOOST_PYTHON_MODULE(boost_pythonSVM)
{
    boost::python::class_<svm>("svm",boost::python::init<int,int,string>())
        .def("train",&svm::train)
        .def("test",&svm::test);
}


/*
g++ boost_pythonSVM.cpp -fPIC -shared -o boost_pythonSVM.so -I/usr/include/python3.8 -I/usr/local/include/boost -L. -lboost_python38 
*/