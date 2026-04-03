function [f,g] = sfun_o(x,probe,dp);

%%===== x is vectorized as elemental channels and imaginary part
global ind_b Np N_obj Ne
global p_overlap
global thres


% thres=1e-10;
probe = probe; % used to be probe = probe + thres

% x = reshape(x, N_obj^2, Ne+1);
% y(1:N_obj^2) = sum(x(:,1:Ne),2);
% y(N_obj^2+1:2*N_obj^2)=x(:,end);

y = x;

obj = realToComplex(y);
f=0;
gz=zeros(N_obj,N_obj);
nx_scan = size(dp,3);


for i=1:nx_scan; 

    obj_roi = obj(ind_b(i,3):ind_b(i,4),ind_b(i,1):ind_b(i,2));

    psi =  obj_roi .* probe;
    psi_old = psi;
    psi = fftshift(fft2(ifftshift(psi)));  
    cmp2=abs(psi);
    % cmp2=abs(psi);
    %%====================data misfit
    res=cmp2-dp(:,:,i);

    % p_overlap_roi=p_overlap(ind_b(i,3):ind_b(i,4),ind_b(i,1):ind_b(i,2));
    % res=res./p_overlap_roi;
    f = f + 0.5*norm(res(:)/(Np^1),2)^2;


    %%======feasibility % useless
    psi = psi./cmp2.*dp(:,:,i);
    psi = fftshift(ifft2(ifftshift(psi)));
    res =psi_old-psi; 

    % res=fftshift(ifft2(ifftshift(fftshift(fft2(ifftshift(psi_old)))./cmp2.*res./p_overlap_roi)));
    sub_g = conj(probe).*res;
    %%==================
    gz(ind_b(i,3):ind_b(i,4),ind_b(i,1):ind_b(i,2))= gz(ind_b(i,3):ind_b(i,4),ind_b(i,1):ind_b(i,2))+sub_g;


end

f=f/nx_scan;
gz=gz/nx_scan;
g(:,1)= real(gz(:));
g(:,2:Ne+1)= repmat(imag(gz(:)), 1, Ne);
g = g(:);





% function [f,g] = sfun_o(x, probe, dp)
% 
% global ind_b Np N_obj Ne
% global p_overlap
% global thres
% global lambda_tv tv_type eps_tv mu_tv
% 
% probe = probe; % (kept as your code)
% y = x;
% obj = realToComplex(y);
% 
% f = 0;
% gz = zeros(N_obj, N_obj);
% nx_scan = size(dp,3);
% 
% eps_amp = 1e-12;  
% 
% for i = 1:nx_scan
% 
%     obj_roi = obj(ind_b(i,3):ind_b(i,4), ind_b(i,1):ind_b(i,2));
%     psi_old = obj_roi .* probe;
%     Psi = fftshift(fft2(ifftshift(psi_old)));
%     amp = abs(Psi);
% 
%     A_meas = dp(:,:,i);  
%     res = amp - A_meas;
%     f = f + 0.5 * norm(res(:) / (Np^1), 2)^2;
% 
%     % amplitude projection
%     Psi_proj = Psi ./ (amp + eps_amp) .* A_meas;
%     psi_new = fftshift(ifft2(ifftshift(Psi_proj)));
% 
%     res_exit = psi_old - psi_new;
%     sub_g = conj(probe) .* res_exit;
% 
%     gz(ind_b(i,3):ind_b(i,4), ind_b(i,1):ind_b(i,2)) = ...
%         gz(ind_b(i,3):ind_b(i,4), ind_b(i,1):ind_b(i,2)) + sub_g;
% 
% end
% 
% f = f / nx_scan;
% gz = gz / nx_scan;
% 
% f_tv = 0;
% gZ_tv = zeros(N_obj, N_obj);
% gO_tv = zeros(N_obj, N_obj);   
% 
% if exist('lambda_tv','var') && ~isempty(lambda_tv) && lambda_tv > 0
% 
%     Zr_img = reshape(y(1:N_obj^2), N_obj, N_obj);
%     Zi_img = reshape(y(N_obj^2+1:2*N_obj^2), N_obj, N_obj);
% 
%     u = Zr_img + 1i * (mu_tv * Zi_img);
%     [tv_val, grad_u] = tv_complex(u, tv_type, eps_tv);
% 
%     f_tv = lambda_tv * tv_val;
%     gZ_tv = lambda_tv * real(grad_u);
%     gO_tv = lambda_tv * (mu_tv * imag(grad_u));
% end
% 
% 
% g_r = real(gz) + gZ_tv;
% g_i = imag(gz) + gO_tv;
% 
% 
% g = zeros(N_obj^2, Ne+1);
% g(:,1) = g_r(:);
% g(:,2:Ne+1) = repmat(g_i(:), 1, Ne);  
% g = g(:);
% 
% f = f + f_tv;
% 
% end