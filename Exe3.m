clc;
clear;
close all;

m = 128;
n = 256;
S = 3:3:63;

num_of_trials = 500;
success_rate_total_omp = zeros(size(S,2),1);
success_rate_total_sp = zeros(size(S,2),1);
success_rate_total_iht = zeros(size(S,2),1);
for Sj = S
for i=1:num_of_trials
    A = randn(m, n);
    A_col_norm = sqrt(sum(A.^2));
    A = A ./ A_col_norm;
    
    % sparse vector
    x = zeros(n,1);
    x_index = randsample(n, Sj);
    x(x_index) = randn(Sj, 1);
    
    y = A * x;
    
    x_hat_omp = omp(Sj, A, y);
    success_rate_total_omp(Sj/3) = success_rate_total_omp(Sj/3) + ((norm(x_hat_omp-x)/norm(x)) < 0.000001);
    x_hat_sp = sp(Sj, A, y);
    success_rate_total_sp(Sj/3) = success_rate_total_sp(Sj/3) + ((norm(x_hat_sp-x)/norm(x)) < 0.000001);
    x_hat_iht = iht(Sj, A, y);
    success_rate_total_iht(Sj/3) = success_rate_total_iht(Sj/3) + ((norm(x_hat_iht-x)/norm(x)) < 0.000001);
end
end
success_rate_omp = 100 * success_rate_total_omp / num_of_trials;
success_rate_sp = 100 *  success_rate_total_sp / num_of_trials;
success_rate_imt = 100 * success_rate_total_iht / num_of_trials;

%%
figure (1);
plot(S, success_rate_omp, S, success_rate_sp, S, success_rate_imt);
title("Success rate of three different algorithms");
legend("OMP", "SP", "IHT");
xlim([0 62]);
ylim([-10 110]);
xlabel("Sparsity (S)");
ylabel("Success Rate (unit: %)");

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
%     disp("Iter: "+i+", Norm: "+yr_norm);
    current_norm = yr_norm;
    if (((current_norm / norm(y)) < 0.000001) || (past_nrom <= current_norm))
    % if ((current_norm / past_nrom) >= 1)
        break;
    end
    past_nrom = current_norm;
end
end