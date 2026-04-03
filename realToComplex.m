%% converts real-valued 2N 

function z = realToComplex(w)
    
% wLength = length(w);
% 
% x = w(1:wLength/2);
% y = w((wLength/2+1):end);

z = w(1:end/2) + 1i*w(end/2+1:end);
N = sqrt(length(z));
z = reshape(z,N,N);
end
