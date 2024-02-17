#ifndef CNN_HPP
#define CNN_HPP

#include <vector>
#include <string>

using namespace std;

class CNN
{
    private:
        double lr{.01};
        double beta1{.95};
        double beta2{.99};
        int numFilter1{8};
        int numFilter2{8};
        vector<vector<int>> params;
        vector<vector<vector<vector<double>>>> f1;
        vector<vector<double>> b1;
        vector<vector<vector<vector<double>>>> f2;
        vector<vector<double>> b2;
        vector<vector<double>> w3;
        vector<vector<double>> b3;
        vector<vector<double>> w4;
        vector<vector<double>> b4;

        void initializeParameters();
        double randGaussian();
        void adamGD(int imageAmount, vector<double> &cost);
        void conv(double &_loss, vector<vector<vector<vector<double>>>> &_df1, vector<vector<vector<vector<double>>>> &_df2, vector<vector<double>> &_dw3, vector<vector<double>> &_dw4, vector<vector<double>> &_db1, vector<vector<double>> &_db2, vector<vector<double>> &_db3, vector<vector<double>> &_db4, vector<vector<vector<double>>> image, vector<vector<double>> label);

    public:
        CNN();

        void train(int epochs, int dataAmount);
        void exportData(string fileName);
        void importData(string fileName);
        void predict(vector<vector<double>> &_probs, vector<vector<vector<double>>> image);
        void getMNISTData(vector<vector<vector<double>>> &d, int &l, int rowNum, string fileName);
};

#endif