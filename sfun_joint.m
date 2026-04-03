function [f,g] = sfun_joint(v,probe,d,dp)

global Np N_obj Ne
global beta1 thres
global ind_b

%%===== v is vectorized as elemental channels and imaginary part
% fluorescence
    O = reshape(v(N_obj^2+1: end), N_obj, N_obj, Ne);
    f1 = 0;
    p = abs(probe);
    p_square = p.^2;


    for i=1:Ne
        r = conv2(O(:,:,i),p_square,'same')-d(:,:,i); 
        f1 = f1+ norm(r(:))^2/2;
        df_I_full=conv2(r,p_square(end:-1:1,end:-1:1),'full');

        nt=floor(size(df_I_full,1)/2);
        ntp=floor(N_obj/2);

        if(mod(N_obj,2)==1)
            df_I(:,:,i)=df_I_full(nt-ntp+1:nt+ntp+1,nt-ntp+1:nt+ntp+1);
        elseif(mod(N_obj,2)==0)
            df_I(:,:,i) = df_I_full(nt-ntp+1:nt+ntp,nt-ntp+1:nt+ntp);
        end

    end

    df_I=df_I(:);
    f1 = f1.*beta1;
    df_I = df_I*beta1;

% ptychography
    probe = probe;

    mu = 1 ; % need to be adjust based on the property of the object
    x = reshape(v, N_obj^2, Ne+1);
    y(1:N_obj^2) = x(:,1);
    y(N_obj^2+1:2*N_obj^2)= mu * O(:);

    % y = v; % for simiplification

    obj = realToComplex(y);
    f2=0;
    gz=zeros(N_obj,N_obj);
    nx_scan = size(dp,3);

    for i=1:nx_scan; 
        obj_roi = obj(ind_b(i,3):ind_b(i,4),ind_b(i,1):ind_b(i,2));

        psi =  obj_roi .* probe;


        psi_old = psi;
        psi = fftshift(fft2(ifftshift(psi)));   
        cmp2=abs(psi);
        % cmp2 = abs(psi);

        %%====================data misfit

        res=cmp2-dp(:,:,i);

        % p_overlap_roi=p_overlap(ind_b(i,3):ind_b(i,4),ind_b(i,1):ind_b(i,2));
        % res=res./p_overlap_roi;

        f2 = f2 + 0.5*norm(res(:)/(Np^1),2)^2;

        %%======feasibility
        psi = psi./cmp2.*dp(:,:,i);
        psi = fftshift(ifft2(ifftshift(psi)));
        res=psi_old-psi;
        % res=fftshift(ifft2(ifftshift(fftshift(fft2(ifftshift(psi_old)))./cmp2.*res./p_overlap_roi)));
        sub_g = conj(probe).*res;
        %%==================
        gz(ind_b(i,3):ind_b(i,4),ind_b(i,1):ind_b(i,2))= gz(ind_b(i,3):ind_b(i,4),ind_b(i,1):ind_b(i,2))+sub_g;
    end

    f2=f2/nx_scan;
    gz=gz/nx_scan;

    g_r=repmat(real(gz(:)),1,Ne);
    g_i=imag(gz(:));

    f = f1 + f2 ;
    g = [g_r(:);g_i(:)+df_I(:)]; % used to be [g_r(:)+df_I(:);g_i(:)]
end


%%
% 
% function [f,g] = sfun_joint(v, probe, d, dp)
% 
% global Np N_obj Ne
% global beta1 thres
% global ind_b
% global lambda_tv tv_type eps_tv mu_tv
% 
% % 1) Fluorescence term
% 
% O = reshape(v(N_obj^2+1:end), N_obj, N_obj, Ne);
% 
% f1 = 0;
% p_square = abs(probe).^2;
% 
% df_I = zeros(N_obj, N_obj, Ne);   % gradient w.r.t O from fluorescence
% 
% for i = 1:Ne
%     r = conv2(O(:,:,i), p_square, 'same') - d(:,:,i);
%     f1 = f1 + 0.5 * norm(r(:))^2;
% 
%     df_I_full = conv2(r, p_square(end:-1:1, end:-1:1), 'full');
% 
%     nt  = floor(size(df_I_full,1)/2);
%     ntp = floor(N_obj/2);
% 
%     if mod(N_obj,2) == 1
%         df_I(:,:,i) = df_I_full(nt-ntp+1:nt+ntp+1, nt-ntp+1:nt+ntp+1);
%     else
%         df_I(:,:,i) = df_I_full(nt-ntp+1:nt+ntp,   nt-ntp+1:nt+ntp);
%     end
% end
% 
% f1   = beta1 * f1;
% df_I = beta1 * df_I;
% 
% mu = 1;  
% Zr_img = reshape(v(1:N_obj^2), N_obj, N_obj);
% O_for_pty = O(:,:,1);
% 
% obj = Zr_img + 1i * (mu * O_for_pty);
% 
% f2 = 0;
% gz = zeros(N_obj, N_obj);
% nx_scan = size(dp,3);
% 
% eps_amp = 1e-12;  % avoid division by zero in projection
% 
% for i = 1:nx_scan
%     obj_roi = obj(ind_b(i,3):ind_b(i,4), ind_b(i,1):ind_b(i,2));
% 
%     psi_old = obj_roi .* probe;
% 
%     Psi = fftshift(fft2(ifftshift(psi_old)));
%     amp = abs(Psi);
%     A_meas = dp(:,:,i);
% 
%     res = amp - A_meas;
%     f2 = f2 + 0.5 * norm(res(:) / (Np^1), 2)^2;
% 
%     % amplitude projection
%     Psi_proj = Psi ./ (amp + eps_amp) .* A_meas;
% 
%     psi_new = fftshift(ifft2(ifftshift(Psi_proj)));
% 
%     res_exit = psi_old - psi_new;
%     sub_g = conj(probe) .* res_exit;
% 
%     gz(ind_b(i,3):ind_b(i,4), ind_b(i,1):ind_b(i,2)) = ...
%         gz(ind_b(i,3):ind_b(i,4), ind_b(i,1):ind_b(i,2)) + sub_g;
% end
% 
% f2 = f2 / nx_scan;
% gz = gz / nx_scan;
% 
% g_r = real(gz);          
% g_i = imag(gz);         
% 
% gO_pty = zeros(N_obj, N_obj, Ne);
% gO_pty(:,:,1) = mu * g_i;   
% 
% 
% % 3) TV regularization on full complex object
% 
% f_tv = 0;
% gZ_tv = zeros(N_obj, N_obj);
% gO_tv = zeros(N_obj, N_obj, Ne);
% 
% if exist('lambda_tv','var') && ~isempty(lambda_tv) && lambda_tv > 0
% 
%     for c = 1:Ne
%         u_c = Zr_img + 1i * (mu_tv * O(:,:,c));
%         [tv_c, grad_u] = tv_complex(u_c, tv_type, eps_tv);
% 
%         f_tv = f_tv + tv_c;
% 
%         gZ_tv = gZ_tv + real(grad_u);
%         gO_tv(:,:,c) = gO_tv(:,:,c) + (mu_tv * imag(grad_u));
%     end
% 
%     f_tv  = lambda_tv * f_tv;
%     gZ_tv = lambda_tv * gZ_tv;
%     gO_tv = lambda_tv * gO_tv;
% end
% 
% f = f1 + f2 + f_tv;
% 
% gZ = g_r + gZ_tv;
% gO = df_I + gO_pty + gO_tv;
% g = [gZ(:); gO(:)];
% 
% end