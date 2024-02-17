#include "CNN.hpp"


CNN::CNN()
{
    params.push_back({numFilter1, 1, 5, 5});          // f1
    params.push_back({numFilter2, numFilter1, 5, 5}); // f2
    params.push_back({128, numFilter2 * 100});        // w3
    params.push_back({10, 128});                      // w4
    params.push_back({numFilter1, 1});                // b1
    params.push_back({numFilter2, 1});                // b2
    params.push_back({128, 1});                       // b3
    params.push_back({10, 1});                        // b4
    f1 = vector<vector<vector<vector<double>>>>(params[0][0], vector<vector<vector<double>>>(params[0][1], vector<vector<double>>(params[0][2], vector<double>(params[0][3], 0))));
    f2 = vector<vector<vector<vector<double>>>>(params[1][0], vector<vector<vector<double>>>(params[1][1], vector<vector<double>>(params[1][2], vector<double>(params[1][3], 0))));
    w3 = vector<vector<double>>(params[2][0], vector<double>(params[2][1], 0));
    w4 = vector<vector<double>>(params[3][0], vector<double>(params[3][1], 0));
    b1 = vector<vector<double>>(params[4][0], vector<double>(params[4][1], 0));
    b2 = vector<vector<double>>(params[5][0], vector<double>(params[5][1], 0));
    b3 = vector<vector<double>>(params[6][0], vector<double>(params[6][1], 0));
    b4 = vector<vector<double>>(params[7][0], vector<double>(params[7][1], 0));
    initializeParameters();
}