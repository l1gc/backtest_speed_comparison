function [position, weight, commission, portfolio_value] = backtest(is_rebalance, returns, target_weight, inital_capital)

    % is_rebalance: t*1的01调仓向量,0代表不调仓
    % returns: t*k的个股收益率矩阵, t日收益率=(t日收盘价-t-1日收盘价)/t-1日收盘价
    % target_weight: t*k的目标权重矩阵,t行代表调仓日t的个股资产权重目标,调仓日外均为0
    % inital_capital: 回测初始资金

    dates = size(target_weight, 1); % dates为回测期间交易日总数
    k = size(target_weight, 2); % k为回测期间个股数

    % 初始化输出矩阵
    position = zeros(dates, k); % 每日收盘后个股资产头寸
    weight = zeros(dates, k); % 每日收盘后个股头寸占总资产权重
    commission = zeros(dates, k); % 每日交易成本

    % 初始化个股头寸
    if is_rebalance(1)
        position(1, :) = inital_capital * target_weight(1, :);
        weight(1, :) = target_weight(1, :);
        commission(1, :) = position(1, :) * 0.0003;
    else
        position(1, :) = repmat(inital_capital / k, 1, k);
        weight(1, :) = repmat(1 / k, 1, k);
        commission(1, :) = position(1, :) * 0.0003;
    end

    for t = 2:dates

        if is_rebalance(t)
            position(t, :) = position(t - 1, :) .* returns(t, :); % t日临收盘时个股资产头寸
            position(t, :) = sum(position(t, :)) * target_weight(t, :); % 调仓
            weight(t, :) = target_weight(t, :);
            commission(t, :) = abs(position(t, :) - position(t - 1, :)) * 0.0003; % 计算调仓成本
            position(t, :) = position(t, :) - commission(t, :); % 调仓完成后个股头寸
        else
            position(t, :) = position(t - 1, :) .* returns(t, :);
            weight(t, :) = position(t, :) ./ sum(position(t, :));
            commission(t, :) = 0;
        end

    end

    portfolio_value = sum(position,2); % 计算t*1的组合头寸

end