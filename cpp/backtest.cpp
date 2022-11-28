#include <cmath>
#include <iostream>
#define EIGEN_USE_BLAS
#define EIGEN_INITIALIZE_MATRICES_BY_ZERO
#include <pybind11/eigen.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <eigen3/Eigen/Dense>
#include <tuple>

using namespace Eigen;
using namespace std;
namespace py = pybind11;

typedef Matrix<double, Dynamic, Dynamic, RowMajor> mat;

std::tuple<mat, mat, mat, mat> backtest(const mat &is_rebalance, const mat &returns, const mat &target_weight, double inital_capital)
{
    // is_rebalance: t*1的01调仓向量,0代表不调仓
    // returns: t*k的个股收益率矩阵, t日收益率=(t日收盘价-t-1日收盘价)/t-1日收盘价
    // target_weight: t*k的目标权重矩阵,t行代表调仓日t的个股资产权重目标,调仓日外均为0
    // inital_capital: 回测初始资金

    int dates = target_weight.rows(); // dates为回测期间交易日总数
    int k = target_weight.cols();     // k为回测期间个股数

    // 提前为输出结果分配内存
    mat position = MatrixXd::Zero(dates, k);        // 每日收盘后个股资产头寸
    mat weight = MatrixXd::Zero(dates, k);          // 每日收盘后个股头寸占总资产权重
    mat commission = MatrixXd::Zero(dates, k);      // 每日交易成本
    mat portfolio_value = MatrixXd::Zero(dates, 1); // 每日资产组合净值

    if (is_rebalance(0, 0) == 1)
    {
        position.row(0) = target_weight.row(0) * inital_capital;
        weight.row(0) = target_weight.row(0);
        commission.row(0) = position.row(0) * 0.0003;
    }
    else
    {
        MatrixXd init_pos(1, 1); 
	init_pos(0,0) = inital_capital / k;
        MatrixXd init_weight(1, 1); 
	init_weight(0, 0) = 1 / k;
        position.row(0) = init_pos.replicate(1, k);
        weight.row(0) = init_weight.replicate(1, k);
        commission.row(0) = position.row(0) * 0.0003;
    }

    for (int t = 1; t < dates; ++t)
    {
        if (is_rebalance(t, 0) == 1)
        {
            position.row(t) = position.row(t - 1).array() * returns.row(t).array(); // t日临收盘时个股资产头寸
            position.row(t) = position.row(t).sum() * target_weight.row(t);         // 调仓
            weight.row(t) = target_weight.row(t);
            commission.row(t) = (position.row(t) - position.row(t - 1)).array().abs() * 0.0003; // 计算调仓成本
            position.row(t) = position.row(t) - commission.row(t);                      //调仓完成后个股头寸
        }
        else
        {
            position.row(t) = position.row(t - 1).array() * returns.row(t).array();
            weight.row(t) = position.row(t) / position.row(t).sum();
            commission.row(t) = MatrixXd::Zero(1, k);
        }
    }

    portfolio_value = position.rowwise().sum();

    return std::make_tuple(position, weight, commission, portfolio_value);
};


PYBIND11_MODULE(mybt, m)
{
    m.doc() = "Simple portfolio backtest";
    m.def("backtest", &backtest, py::return_value_policy::reference_internal);
}