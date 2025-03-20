function X = bdiag_inv(Xbdiag, dims)
% bdiag_inv: 将对角块矩阵还原为三维张量
%
%   X = bdiag_inv(Xbdiag, dims)
%
% 输入参数：
%   Xbdiag - 对角块矩阵，其尺寸为 (n1*n3) x (n2*n3)
%   dims   - 一个向量 [n1, n2, n3]，表示原始张量的尺寸
%
% 输出参数：
%   X      - 还原后的三维张量，其尺寸为 n1 x n2 x n3
%
% 示例：
%   X = rand(4,3,5);
%   Xbdiag = bdiag(X);
%   X_recovered = bdiag_inv(Xbdiag, [4,3,5]);

n1 = dims(1);
n2 = dims(2);
n3 = dims(3);

X = zeros(n1, n2, n3);
for i = 1:n3
    % 从 Xbdiag 中提取第 i 个对角块
    row_idx = (i-1)*n1 + 1 : i*n1;
    col_idx = (i-1)*n2 + 1 : i*n2;
    X(:,:,i) = Xbdiag(row_idx, col_idx);
end
end
