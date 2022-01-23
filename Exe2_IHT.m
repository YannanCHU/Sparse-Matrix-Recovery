clc;
clear;
close all;

m = 128;
n = 256;
S = 12;

A = randn(m, n);
% the Gaussian distribution is used, the correlation between ai and aj
% should be zero. Thus, this randn matix can be approximately considered as
% orthogonal matrix.
% column normalization
A_col_norm = sqrt(sum(A.^2));
A = A ./ A_col_norm;
% A_col_norm2 = sqrt(sum(A.^2));

% sparse vector
x = zeros(n,1);
x_index = randsample(n, S);
x(x_index) = randn(S, 1);

y = A * x;
x_hat_1 = pinv(A) * y;
x_hat_2 = A\y;

% IHT
% Initialization
x_l = zeros(n,1);
past_nrom = norm(y-A*x_l, 2);

for i = 1:1000
    temp1 = x_l + A' * (y-A*x_l);
    [temp1_sorted, temp1_pos] = sort(abs(temp1), 'descend');
    Supp = sort(temp1_pos(1:S));
    x_l = zeros(n, 1);
    x_l(Supp, 1) = temp1(Supp, 1);
    yr_norm = norm(y-A*x_l, 2);
    disp("Iter: "+i+", Norm: "+yr_norm);
    current_norm = yr_norm;
    if (((current_norm / norm(y)) < 0.000001) || (past_nrom <= current_norm))
    % if ((current_norm / past_nrom) >= 1)
        break;
    end
    past_nrom = current_norm;
    
end
