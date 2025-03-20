% applying tensor completion for image inpainting

clear
X = double(imread('testimg.jpg'));
X = X/255;
search_r=40;
tubal_r=search_r;

[u1,s1,v1]=tsvd_r(X,tubal_r);

X=tprod(tprod(u1,s1),tran(v1));

maxP = max(abs(X(:)));
[n1,n2,n3] = size(X);

p = 0.5; % sampling rate

X_star=X;
omega_seed=rand(n1,n2,n3);
omega=omega_seed < p;
Y=omega.*X;

opts.DEBUG = 1;


%% initialization


[U_ini,S_ini,V_ini]=tsvd_r(Y/p,40);
Lt=tprod(U_ini,sqrt(S_ini));
Rt=tprod(V_ini,sqrt(S_ini));
Xt=tprod(Lt,tran(Rt));

norm(X_star - Xt, 'fro')/norm(X_star, 'fro')


%% PreGD
ite=100;
error_PreGD=zeros(ite,1);
time_PreGD=zeros(ite,1);
mu=0.7;
Ir=teye(search_r,n3);
tCount1=0;
tic
for i=1:ite
    t1=tic;
    Xt=tprod(Lt,tran(Rt));
    error_PreGD(i) = norm(X_star - Xt, 'fro')/norm(X_star, 'fro');
    Zt=omega.*Xt;
    Lt1=Lt-mu*tprod((tprod(Zt,Rt)-tprod(Y,Rt)),tinv(tprod(tran(Rt),Rt)));
    Rt1=Rt-mu*tprod(tprod(tran(Zt-Y),Lt),tinv(tprod(tran(Lt),Lt)));
    Zt1=omega.*tprod(Lt1,tran(Rt1));
    error=norm(Zt-Zt1,'fro')/norm(Y(:))
     if error < 1e-3
            Zt=omega.*Xt;
            Lt1=Lt-mu*tprod((tprod(Zt,Rt)-tprod(Y,Rt)),tinv(tprod(tran(Rt),Rt)));
            Rt1=Rt-mu*tprod(tprod(tran(Zt-Y),Lt),tinv(tprod(tran(Lt),Lt)));
            Lt=Lt1;Rt=Rt1;
            tCount1=tCount1+toc(t1);
            time_PreGD(i)=tCount1;
        break;
    end
    Lt=Lt1;Rt=Rt1;
    tCount1=tCount1+toc(t1);
    time_PreGD(i)=tCount1;
end
toc



Xhat=Xt;

Xhat = max(Xhat,0);
Xhat = min(Xhat,maxP);
psnr = PSNR(X,Xhat,maxP)


figure(1)
subplot(1,3,1)
imshow(X)
subplot(1,3,2)
imshow(Y)
subplot(1,3,3)
imshow(Xhat)

function [U,S,V] = tsvd_r(X,tubal_r)

[n1,n2,n3] = size(X);
X = fft(X,[],3);
U = zeros(n1,tubal_r,n3);
S = zeros(tubal_r,tubal_r,n3);
V = zeros(n2,tubal_r,n3);
    
% i=1 
[U(:,:,1),S(:,:,1),V(:,:,1)] = svds(X(:,:,1),tubal_r);

halfn3 = round(n3/2);
for i = 2 : halfn3
    [U(:,:,i),S(:,:,i),V(:,:,i)] = svds(X(:,:,i),tubal_r);
    U(:,:,n3+2-i) = conj(U(:,:,i));
    V(:,:,n3+2-i) = conj(V(:,:,i));
    S(:,:,n3+2-i) = S(:,:,i);
end    
% if n3 is even
if mod(n3,2) == 0
    i = halfn3+1;
    [U(:,:,i),S(:,:,i),V(:,:,i)] = svds(X(:,:,i),tubal_r);
end

for i = 1 : n3
    [U_temp,S_temp,V_temp] = svds(X(:,:,i),tubal_r);       
    U(:,:,i) = U_temp;
    V(:,:,i) = V_temp;
    S(:,:,i) = S_temp;
end


U = ifft(U,[],3);
S = ifft(S,[],3);
V = ifft(V,[],3);
end
