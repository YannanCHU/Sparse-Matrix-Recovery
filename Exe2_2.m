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

% Three Greedy Algorithms
x_hat_omp = omp(S, A, y);
x_hat_sp = sp(S, A, y);
x_hat_iht = iht(S, A, y);

%% Comparison
figure(1);
plot(1:n, x, 1:n, x_hat_1, 1:n, x_hat_2, 1:n, x_hat_omp, 1:n, x_hat_sp, 1:n, x_hat_iht);
legend("x", "x hat 1", "x hat 2", "OMP", "SP", "IHT");

figure(2);
plot(1:n, x, 1:n, x_hat_sp);
legend("x", "SP");

function x_hat_omp = omp(S, A, y)
% Initialization
n = size(A,2);
x_hat_omp = zeros(n,1);
yr = y;
Supp = [];

for L = 1:S
    x_appro = A' * yr;
    [R,~] = find(x_appro == max(x_appro));
    Supp = [Supp R];
    
    As = A(:, Supp);
    x_hat_s = As \ y;
    
    x_hat_omp(Supp,1) = x_hat_s;
    
    yr = y-A*x_hat_omp;
end
end

% SP
function x_hat_sp = sp(S, A, y)
% Initialization
n = size(A,2);
x_hat_sp = zeros(n,1);
% supp(Hs(A'*y))
[~,pos] = sort(abs(A' * y), 'descend');      Supp = pos(1:S);
As = A(:, Supp);
yp = As * (As \ y);
yr = y - yp;
past_nrom = norm(yr);

Supp_bar = [];
% Iteration
for i = 1:200
    [~,pos] = sort(abs(A' * yr), 'descend');      Supp2 = pos(1:S);
    Supp_bar = [Supp; Supp2];
    
    A_s_bar = A(:, Supp_bar);
    b_s_bar = pinv(A_s_bar) * y;
    b = zeros(n,1);
    b(Supp_bar,1) = b_s_bar;
    
    % step 3: supp(Hs(b))
    [~,pos] = sort(abs(b), 'descend');      Supp = pos(1:S);
    % step 4
    x_hat_sp = zeros(n,1);
    x_hat_sp(Supp, 1) = A(:, Supp) \ y;
    % step 5
    yr = y - A * x_hat_sp;
    current_norm = norm(yr);
    
    % disp("Iter: " + i + ", " + (norm(yr) / norm(y)));
    
    if (((current_norm / norm(y)) < 0.000001) || (past_nrom <= current_norm))
    % if ((current_norm / past_nrom) >= 1)
        break;
    end
    
    past_nrom = current_norm;
end
end

% IHT
function x_l = iht(S, A, y)
% Initialization
n = size(A,2);
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
end
