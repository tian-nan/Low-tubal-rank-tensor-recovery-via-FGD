% Low-Rank Tensor Completion (LRTC)

clear

n = 100;
n1 = n;
n2 = n;
n3 = 10;
r = 5; % tubal rank
X = tprod(randn(n1,r,n3)/n1,randn(r,n2,n3)/n2); % size: n1*n2*n3


p = 0.8;

omega = find(rand(n1*n2*n3,1)<p);
M = zeros(n1,n2,n3);
M(omega) = X(omega);


opts.DEBUG = 1;
tic
Xhat = lrtc_tnn(M,omega,opts);
toc
trank = tubalrank(Xhat);
RSE = norm(X(:)-Xhat(:))/norm(X(:));

fprintf('\nsampling rate: %f\n', p);
fprintf('tubal rank of the underlying tensor: %d\n',r);
fprintf('tubal rank of the recovered tensor: %d\n', trank);
fprintf('relative recovery error: %.4e\n', RSE);


