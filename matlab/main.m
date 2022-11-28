weig=csvread("weights.csv");
returns=csvread("returns.csv");
isr=csvread("isr.csv");

tic
init=1000;
[position, weight, commission, portfolio_value] = backtest(isr, returns, weig, init);

toc