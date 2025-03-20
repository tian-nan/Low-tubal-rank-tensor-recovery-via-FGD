function F_0 = tsvd_ini(X,tubal_r)

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
