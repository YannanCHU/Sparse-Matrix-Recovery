clc;
clear;
close all;

m = 128;
n = 256;

A = randn(m, n);
% column normalization
A_col_norm = sqrt(sum(A.^2));
A = A ./ A_col_norm;
A_col_norm2 = sqrt(sum(A.^2));

x = randn(n,1);
y = A * x;

x_hat_1 = pinv(A) * y;
x_hat_2 = A\y;
x_hat_3 = (inv(A'*A) * A') * y;

mse1 = sum((x_hat_1-x).^2) / size(x,1)
mse2 = sum((x_hat_2-x).^2) / size(x,1)

plot(1:n, x, 1:n, x_hat_1, 1:n, x_hat_2);   xlim([0 260]);
legend("x", "x hat 1 - pinv(A) * y with MSE: " + mse1, "x hat 2 - A\y with MSE: " + mse2);
title("Comparison between pinv(A)*y and A\y methods");


