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

void CNN::initializeParameters()
{
    srand(time(NULL));
    double dev1 = 1 / sqrt(f1.size() * f1[0].size() * f1[0][0].size() * f1[0][0][0].size());
    double dev2 = 1 / sqrt(f2.size() * f2[0].size() * f2[0][0].size() * f2[0][0][0].size());
    for (int i = 0; i < f1.size(); i++)
    {
        for (int j = 0; j < f1[0].size(); j++)
        {
            for (int k = 0; k < f1[0][0].size(); k++)
            {
                for (int l = 0; l < f1[0][0][0].size(); l++)
                {
                    double f = (double)rand() / RAND_MAX;
                    double r = f * 4 - 2;
                    f1[i][j][k][l] = dev1 * exp(-1 * ((r * r) / 2));
                    if (((double)rand() / (RAND_MAX)) > 0.5)
                    {
                        f1[i][j][k][l] *= -1;
                    }
                    else
                    {
                        f1[i][j][k][l] *= 1;
                    }
                }
            }
        }
    }
    for (int i = 0; i < f2.size(); i++)
    {
        for (int j = 0; j < f2[0].size(); j++)
        {
            for (int k = 0; k < f2[0][0].size(); k++)
            {
                for (int l = 0; l < f2[0][0][0].size(); l++)
                {
                    double f = (double)rand() / RAND_MAX;
                    double r = f * 6 - 3;
                    f2[i][j][k][l] = dev2 * exp(-1 * ((r * r) / 2));
                    if (((double)rand() / (RAND_MAX)) > 0.5)
                    {
                        f2[i][j][k][l] *= -1;
                    }
                    else
                    {
                        f2[i][j][k][l] *= 1;
                    }
                }
            }
        }
    }
    for (int i = 0; i < w3.size(); i++)
    {
        for (int j = 0; j < w3[0].size(); j++)
        {
            w3[i][j] = randGaussian() * 0.01;
        }
    }
    for (int i = 0; i < w4.size(); i++)
    {
        for (int j = 0; j < w4[0].size(); j++)
        {
            w4[i][j] = randGaussian() * 0.01;
        }
    }
}