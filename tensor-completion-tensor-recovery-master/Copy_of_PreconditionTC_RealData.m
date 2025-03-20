% applying tensor completion for image inpainting

clear
X = double(imread('testimg.jpg'));
X = X/255; 
search_r=40;
tubal_r=search_r;
X1=X;
[u1,s1,v1]=tsvd_r(X,tubal_r);
X=tprod(tprod(u1,s1),tran(v1));

% norm(X1-X,'fro')/norm(X1(:))
[psnr_ADMM,t_ADMM,psnr_GD,t_GD,psnr_PreGD,t_PreGD]=TC(X,tubal_r)








function [psnr_ADMM,t_ADMM,psnr_GD,t_GD,psnr_PreGD,t_PreGD]=TC(X,tubal_r)
maxP = max(abs(X(:)));
[n1,n2,n3] = size(X);
p = 0.5; % sampling rate
omega_seed=rand(n1,n2,n3);
omega=omega_seed < p;
Y=omega.*X;

%% ADMM
omega1 = find(omega_seed<p);
M = zeros(n1,n2,n3);
M(omega1) = X(omega1);
M2 = Frontal2Lateral(M); % each lateral slice is a channel of the image
omega2 = zeros(n1,n2,n3);
Iones = ones(n1,n2,n3);
omega2(omega) = Iones(omega);
omega2 = Frontal2Lateral(omega2);
omega2 = find(omega2==1);

opts.DEBUG = 1;
t1=tic;
X_ADMM = lrtc_tnn(M2,omega2,opts);
t_ADMM=toc(t1);
X_ADMM = max(X_ADMM,0);
X_ADMM = min(X_ADMM,maxP);
X_ADMM = Lateral2Frontal(X_ADMM); % each lateral slice is a channel of the image
psnr_ADMM = PSNR(X,X_ADMM,maxP);


%% PreGD
%% initialization


[U_ini,S_ini,V_ini]=tsvd_r(Y/p,tubal_r);
Lt=tprod(U_ini,sqrt(S_ini));
Rt=tprod(V_ini,sqrt(S_ini));
Xt=tprod(Lt,tran(Rt));
X_star=X;
% norm(X_star - Xt, 'fro')/norm(X_star, 'fro');


%% PreGD
ite=100;
error_PreGD=zeros(ite,1);
mu=0.7;
t2=tic;
for i=1:ite
    Xt=tprod(Lt,tran(Rt));
    error_PreGD(i) = norm(X_star - Xt, 'fro')/norm(X_star, 'fro');
    Zt=omega.*Xt;
    Lt1=Lt-mu*tprod((tprod(Zt,Rt)-tprod(Y,Rt)),tinv(tprod(tran(Rt),Rt)));
    Rt1=Rt-mu*tprod(tprod(tran(Zt-Y),Lt),tinv(tprod(tran(Lt),Lt)));
    Zt1=omega.*tprod(Lt1,tran(Rt1));
    error=norm(Zt-Zt1,'fro')/norm(Y(:));
     if error < 1e-3
        break;
    end
    Lt=Lt1;Rt=Rt1;
end
t_PreGD=toc(t2);

X_PreGD=Xt;
X_PreGD = max(X_PreGD,0);
X_PreGD = min(X_PreGD,maxP);
psnr_PreGD = PSNR(X,X_PreGD,maxP);


%% GD
%% initialization


[U_ini,S_ini,V_ini]=tsvd_r(Y/p,tubal_r);
Lt=tprod(U_ini,sqrt(S_ini));
Rt=tprod(V_ini,sqrt(S_ini));
Xt=tprod(Lt,tran(Rt));

% norm(X_star - Xt, 'fro')/norm(X_star, 'fro');


%% GD
ite=500;
error_PreGD=zeros(ite,1);
mu=0.003;
t3=tic;
for i=1:ite
    Xt=tprod(Lt,tran(Rt));
    error_PreGD(i) = norm(X_star - Xt, 'fro')/norm(X_star, 'fro');
    Zt=omega.*Xt;
    Lt1=Lt-mu*(tprod(Zt,Rt)-tprod(Y,Rt));
    Rt1=Rt-mu*tprod(tran(Zt-Y),Lt);
    Zt1=omega.*tprod(Lt1,tran(Rt1));
    error=norm(Zt-Zt1,'fro')/norm(Y(:));
     if error < 1e-4 || error>1e3
        break;
    end
    Lt=Lt1;Rt=Rt1;
end
t_GD=toc(t3);

X_GD=Xt;
X_GD = max(X_GD,0);
X_GD = min(X_GD,maxP);
psnr_GD = PSNR(X,X_GD,maxP);

end

