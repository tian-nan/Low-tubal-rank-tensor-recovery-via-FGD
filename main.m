%% Main Script for Tensor Completion Comparison
% Compare Gradient Descent (GD) and Tensor Nuclear Norm (TNN) methods
% for low-rank tensor completion under Gaussian measurements

clc;clear;close all;
addpath(genpath('nonconvex_funs'));
addpath(genpath('tensor-completion-tensor-recovery-master'));


%% Experiment Parameters
tubal_r = 3;     % True tubal rank of the tensor
k = 3;           % Over-estimated rank for GD initialization
n1 = 50;         % Tensor dimension 1
n2 = n1;         % Tensor dimension 2
n3 = 3;          % Tensor dimension 3 (tubal dimension)
ite = 5e2;       % Maximum number of iterations
repeat_time = 10; % Number of experiment repetitions
%% Initialize Result Containers
error = zeros(ite, 3, repeat_time);
time = zeros(ite, 3, repeat_time);

%% Run Experiments
for i=1:repeat_time
    [error_temp,time_temp]=LRTR(n1,n2,n3,tubal_r,k,ite);
    error(:,:,i)=error_temp;
    time(:,:,i)=time_temp;
end

%% Process Results
% Gradient Descent Results
error_GD_mean=mean(squeeze(error(:,1,:)),2)';
time_GD=squeeze(time(:,1,1));
error_GD=squeeze(error(:,1,1));
time_GD_mean=(squeeze(time(:,1,1)));

% TNN Results
error_TNN_mean=mean(squeeze(error(:,2,:)),2)';
error_TNN=squeeze(error(:,2,1));
time_TNN_mean=squeeze(time(:,2,1));
time_TNN=squeeze(time(:,2,1));


x=1:ite;

figure(1)
p1 = plot(x,error_GD_mean,LineWidth=2,Color=[189,30,30]/255);
hold on;
p2 = plot(x,error_TNN_mean,LineWidth=2,Color=[135 207 164]/255);
hold on;

set(gca,'yscale','log');
legend([p1,p2],'GD','TNN');
xlabel('Iterations');
ylabel('Relative Error');

figure(2)
p1 = plot(time_GD,error_GD,LineWidth=2,Color=[189,30,30]/255);
hold on;
p2 = plot(time_TNN,error_TNN,LineWidth=2,Color=[135 207 164]/255);
hold on;

set(gca,'yscale','log');
legend([p1,p2],'GD','TNN');
xlabel('Clock time');
ylabel('Relative Error');


function [error,time]=LRTR(n1,n2,n3,tubal_r,k,ite)

%% init X_*, A_i and y
threshold=1e-10;
m=10*tubal_r*(n1+n2-tubal_r)*n3;

%% Generate Synthetic Data
% Create true low-rank tensor
U_star=randn(n1,tubal_r,n3);
X_star=tprod(U_star,tran(U_star));

% Measurement matrix and noise
A=normrnd(0,sqrt(1/m),m,n1*n2*n3);
s=normrnd(0,0.01,m,1); % noise
y=A*X_star(:);


%% Method 1: Gradient Descent Approach
%  initialization
t_GD = tic;

F_0 = randn(n1,k,n3);

Xt=tprod(F_0,tran(F_0));

Ft = F_0;
% gradient descent
mu=2e-3;
error_GD=zeros(ite,1);
time_GD=zeros(ite,1);
tcount=0;

for i=1:ite
    t1=tic;

    Xt=tprod(Ft,tran(Ft));
    res=A*reshape(Xt,[n1*n1*n3,1])-y;
    A_star=reshape(A'*res,[n1,n1,n3]);
    Ft1=Ft-mu*(tprod(A_star,Ft));
    Ft=Ft1;
    error_GD(i)=norm(X_star-Xt,'fro')/norm(X_star,'fro');
    if ~isfinite(error_GD(i)) || error_GD(i) > 1e3 || error_GD(i) < threshold
        tcount=tcount+toc(t1);
        time_GD(i)=tcount;
    break;
    end
    tcount=tcount+toc(t1);
    time_GD(i)=tcount;
end
toc(t_GD);
RelErr = norm(X_star(:)-Xt(:),'fro')/norm(X_star(:),'fro');


%% Method 2: Tensor Nuclear Norm (TNN) Approach
opts.error=threshold;
opts.ite=ite;
Xsize.n1 = n1;
Xsize.n2 = n2;
Xsize.n3 = n3;
t_TNN=tic;
[Xhat,error_TNN,time_TNN] = lrtr_Gaussian_tnn(A,y,X_star,Xsize,opts);
toc(t_TNN)
RelErr = norm(X_star(:)-Xhat(:),'fro')/norm(X_star(:),'fro')

%% IRTNN
fun = 'lp';         gamma = 0.5;
lambda = 1e-1;
sizeX=[n1,n2,n3];
t_IRTNN=tic;
error_IRTNN = zeros(ite,1);
time_IRTNN = zeros(ite,1);
% [XRec,error_IRTNN,time_IRTNN]= IRTNN(fun,y,A,sizeX,gamma,lambda,X_star,ite,threshold);
toc(t_IRTNN)
% RelErr = norm(X_star(:)-XRec(:),'fro')/norm(X_star(:),'fro')

error=[error_GD(1:ite),error_TNN,error_IRTNN(1:ite)];
time=[time_GD(1:ite),time_TNN,time_IRTNN(1:ite)];
end





