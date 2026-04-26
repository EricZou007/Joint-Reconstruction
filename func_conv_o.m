% function [f,df_I]=func_conv_o(I,p,d)
% global Np N_obj Ne
% global beta1 thres
% 
% 
%     O = reshape(I, N_obj, N_obj, Ne);
%     f = 0;
%     n = N_obj-Np + 1;
%     n_start=floor(N_obj/2-n/2);
%     % p = abs(p) +thres;
%     p = abs(p);
%     p_square = p.^2;
% 
%     for i=1:Ne
%         r = conv2(O(:,:,i),p_square,'same')-d(:,:,i);
%         f = f+ norm(r(:))^2/2;
% 
%         df_I_full=conv2(r,p_square(end:-1:1,end:-1:1),'full');
% 
%         nt=floor(size(df_I_full,1)/2);
%         ntp=floor(N_obj/2);
%         if(mod(N_obj,2)==1)
%             df_I(:,:,i)=df_I_full(nt-ntp+1:nt+ntp+1,nt-ntp+1:nt+ntp+1);
%         elseif(mod(N_obj,2)==0)
%             df_I(:,:,i) = df_I_full(nt-ntp+1:nt+ntp,nt-ntp+1:nt+ntp);
%         end
%         % df_I(:,:,i)=conv2(r,p(end:-1:1,end:-1:1),'same');
%         % df_I(:,:,i)=conv2(r(n_start+1:n_start+n,n_start+1:n_start+n),p(end:-1:1,end:-1:1),'full');
%     end
% 
%     df_I=df_I(:);
%     f =   f;
%     df_I = df_I;

 
%%


function [f, df_I] = func_conv_o(I, p, d)

global Np N_obj Ne
global beta1 thres
global lambda_tv tv_type eps_tv mu_tv   % TV globals

O = reshape(I, N_obj, N_obj, Ne);

f = 0;
p = abs(p);
p_square = p.^2;

df_I = zeros(N_obj, N_obj, Ne);

for i = 1:Ne
    r = conv2(O(:,:,i), p_square, 'same') - d(:,:,i);
    f = f + 0.5 * norm(r(:))^2;

    df_I_full = conv2(r, p_square(end:-1:1, end:-1:1) , 'full');

    nt  = floor(size(df_I_full,1)/2);
    ntp = floor(N_obj/2);

    if mod(N_obj,2) == 1
        df_I(:,:,i) = df_I_full(nt-ntp+1:nt+ntp+1, nt-ntp+1:nt+ntp+1);
    else
        df_I(:,:,i) = df_I_full(nt-ntp+1:nt+ntp,   nt-ntp+1:nt+ntp);
    end
end


f_tv  = 0;
g_tvO = zeros(N_obj, N_obj, Ne);

if exist('lambda_tv','var') && ~isempty(lambda_tv) && lambda_tv > 0
    for c = 1:Ne
        u = O(:,:,c);

        [tv_val, grad_u] = tv_complex(u, tv_type, eps_tv);

        f_tv = f_tv + tv_val;
        g_tvO(:,:,c) = g_tvO(:,:,c) + real(grad_u);
    end

    f_tv  = lambda_tv * f_tv;
    g_tvO = lambda_tv * g_tvO;
end

f    = f + f_tv;
df_I = df_I + g_tvO;

df_I = df_I(:);

end