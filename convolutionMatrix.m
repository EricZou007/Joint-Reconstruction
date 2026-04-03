function [P,P_full] = convolutionMatrix(p,n,m)

% !!! FOR NE=1 !!! %

% Input
% p     : probe of size m x m
% n     : size of object
% m     : size of probe

% Output
% P     : matrix representation of "same" convolution
% P_full: matrix representation of "full" convolution


P_full = convmtx2(p,n,n);
P = []; 

if mod(m,2)==1
    mb = (m-1)/2;
    mt = mb;
elseif mod(m,2)==0
    mb = floor((m-1)/2);
    mt = mb + 1;
end
start = (n+mb+mt)*mt + mt + 1;

% extract "same" from "full" matrix
for j=1:n
    P((j-1)*n+1:j*n,:) = P_full(start:start+n-1,:);
    start = start + n + mb + mt;
end

end