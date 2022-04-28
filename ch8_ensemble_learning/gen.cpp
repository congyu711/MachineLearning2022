#include<bits/stdc++.h>
using namespace std;

int main()
{
    mt19937 gen;
    ofstream fout("adaboost.data");
    fout<<"h w g\n";
    for(int i=1;i<25;i++)
    {
        fout<<170+gen()%30<<' '<<65+gen()%20<<" 1\n";
    }
    for(int i=1;i<25;i++)
    {
        fout<<160+gen()%30<<' '<<55+gen()%20<<" -1\n";
    }
}