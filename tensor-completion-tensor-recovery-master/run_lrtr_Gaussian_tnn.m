% Low rank tensor recovery from Gaussian measurements
clear

n1 = 10;
n2 = n1; 
n3 = 5;
r = 3; % tubal rank
tubal_r=r;
U_star=zeros(n1,tubal_r,n3);
V_star=zeros(n2,tubal_r,n3);
for i=1:n3
% multi_r=randi([1,tubal_r],1,1);
multi_r=tubal_r;
U_star(:,1:multi_r,i)=orth(randn(n1,multi_r));
V_star(:,1:multi_r,i)=orth(randn(n1,multi_r));
end
X_star=tprod(U_star,tran(U_star));




X=X_star;
% X = tprod(randn(n1,r,n3),randn(r,n2,n3)); % size: n1*n2*n3

m = 3*r*(n1+n2-r)*n3; % number of measurements
n = n1*n2*n3;
A = randn(m,n)/sqrt(m);

b = A*X(:);
Xsize.n1 = n1;
Xsize.n2 = n2;
Xsize.n3 = n3;

opts.DEBUG = 1;
tic
[Xhat,error] = lrtr_Gaussian_tnn(A,b,X_star,Xsize,opts);
toc
RSE = norm(Xhat(:)-X(:))/norm(X(:));
trank = tubalrank(Xhat);

fprintf('\ntubal rank of the underlying tensor: %d\n',r);
fprintf('tubal rank of the recovered tensor: %d\n', trank);
fprintf('number of mesurements: %d\n', m);
fprintf('relative recovery error: %.4e\n', RSE);

