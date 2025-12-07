#include <iostream>
#include <vector>
#include <algorithm>
#include <limits>

using namespace std;

const double INF = 1e9;

class KuhnMunkres{
public:
    void solve(const vector<vector<double>>& W, vector<int>& matchX_, vector<int>& matchY_, double threshold = INF){
        if(W.empty()) return;
        n = W.size();
        m = W[0].size();
        N = max(n, m);
        w = W;
        for(auto& x : w){
            for(auto& y : x) y = -y;
            x.resize(N, -INF);
        }
        w.resize(N, vector<double>(N, -INF));

        lx.assign(N, 0);
        ly.assign(N, 0);
        for(int i = 0; i < N; i++){//把各个lx的值都设为当前w[i][j]的最大值
            lx[i] = -INF;
            for(int j = 0; j < N; j++){
                if(lx[i] < w[i][j])
                    lx[i] = w[i][j];
            }
        }
        matchY.assign(N, -1);
        for(int i = 0; i < N; i++){
            while(1){
                visx.assign(N, 0);
                visy.assign(N, 0);
                if(dfs(i))//如果它能够形成一条增广路径，那么就break
                    break;
                double d = INF;//否则，后面应该加入新的边,这里应该先计算d值
                for(int j = 0; j < N; j++)//对于搜索过的路径上的XY点，设该路径上的X顶点集为S，Y顶点集为T，对所有在S中的点xi及不在T中的点yj
                    if(visx[j])
                        for(int k = 0; k < N; k++)
                        if(!visy[k])
                            d = min(d, lx[j] + ly[k] - w[j][k]);
                if(d == INF)
                    break;  //找不到可以加入的边，返回失败（即找不到完美匹配）
                for (int j = 0; j < N; j++)
                    if (visx[j]) lx[j] -= d;
                for (int j = 0; j < N; j++)
                    if (visy[j]) ly[j] += d;
            }
        }
        matchX.assign(N, -1);
        for(int i = 0; i < N; i++){
            if(~matchY[i]){
                if(-w[matchY[i]][i] < threshold){
                    matchX[matchY[i]] = i;
                }else{
                    matchY[i] = -1;
                }
            }
        }
        matchX.resize(n);
        matchY.resize(m);
        matchX_ = matchX;
        matchY_ = matchY;
    }

private:
    int n, m, N;
    vector<double> lx, ly;
    vector<int> matchX, matchY, visx, visy;
    vector<vector<double>> w;

    int dfs(int t){
        visx[t] = 1;
        for(int i = 0; i < N; i++){
            if(!visy[i] && abs(lx[t] + ly[i] - w[t][i]) < 1e-6){//这里“lx[t]+ly[i]==w[t][i]”决定了这是在相等子图中找增广路的前提，非常重要
                visy[i] = 1;
                if(matchY[i] == -1 || dfs(matchY[i])){
                    matchY[i] = t;
                    return 1;
                }
            }
        }
        return 0;
    }
};
