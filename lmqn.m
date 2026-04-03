function [xstar, infos, ierror, eig_values] = ...
    lmqn (x, sfun, maxit, maxfun, stepmx, accrcy, v_o_true, eig_val_need, separate_loss)

%---------------------------------------------------------
% truncated-newton method for unconstrained minimization
% (customized version)
%---------------------------------------------------------
global hyk ykhyk hyr yksr ykhyr yrhyr sk yk sr yr ...
    hg gsk yksk yrsr
global NF N current_n 
global gv ptest maxiter
global Pw H_init p
global N_obj
global obj N_scan ind_b
global fctn_o
global fctn_f
global w_true
%---------------------------------------------------------
% set up
%---------------------------------------------------------
format compact
format short e
fprintf(1,'  it     nf     cg           f             |g|\n')
upd1 = 1;
ncg  = 0;
nind = find(N==current_n);
xnorm  = norm(x,2); % use to be inf
ierror = 0;
if (stepmx < sqrt(accrcy) | maxfun < 1);
    ierror = -1;
    xstar = x;
    return;
end;
%---------------------------------------------------------
% compute initial function value and related information
%---------------------------------------------------------

[f, g] = feval (sfun, x);



oldf   = f;
gnorm  = norm(g,'inf');



nf     = 1;
nit    = 0;
%%==============store iterative info
infos.epoch = nit;
infos.grad_calc_count = nf+ncg;      
infos.cost = f;
infos.gnorm = gnorm;
%%=================================
fprintf(1,'%4i   %4i   %4i   % .8e   %.1e\n', ...
    nit, nf, ncg, f, gnorm)
%---------------------------------------------------------
% check for small gradient at the starting point.
%---------------------------------------------------------
ftest = 1 + abs(f);
if (gnorm < .01*sqrt(eps)*ftest * 1e-16);
    ierror = 0;
    xstar = x;
    NF(1,nind) = NF(1,nind) + nit;
    NF(2,nind) = NF(2,nind) + nf;
    NF(3,nind) = NF(3,nind) + ncg;
    return;
end;
%---------------------------------------------------------
% set initial values to other parameters
%---------------------------------------------------------
n      = length(x);
icycle = n-1;
toleps = sqrt(accrcy) + sqrt(eps);
rtleps = accrcy + eps;
difnew = 0;
epsred = .05;
fkeep  = f;
conv   = 0;
ireset = 0;
ipivot = 0;



%---------------------------------------------------------
% initialize diagonal preconditioner to the identity
%---------------------------------------------------------



%% initialize the preconditioner


if strcmp(H_init, 'probe-diag')
    L = 50;   % e.g., 5 probe vectors
    n = length(x);
    d_est = zeros(n,1);
    
    for ell = 1:L
        z = sign(randn(n,1));           % ±1 Rademacher
        w = gtims(z, x, g, accrcy, xnorm, sfun);  % w = H_k z  (your existing matvec)
        d_est = d_est + z .* w;
    end
    
    d_est = d_est / L;
    eps_diag = 1e-8;
    d = 1 ./ max(abs(d_est), eps_diag);   % inverse diag for M_k^{-1}

elseif strcmp(H_init, 'jacobi')
    % Build Jacobi preconditioner from approximate Hessian at x
    H_approx = approximate_hessian(x, g, accrcy, xnorm, sfun);
    H_approx = 0.5 * (H_approx + H_approx');      % enforce symmetry

    d = diag(H_approx);                           % Jacobi: diag(H)
    % Safeguard: avoid zero or negative diagonals
    tol_diag = 1e-8;
    idx_bad  = (abs(d) < tol_diag);
    d(idx_bad) = tol_diag;

elseif strcmp(H_init, 'standard')
    d = ones(n,1);

else
    error('Unknown H_init option: %s', H_init);
end

%%



%---------------------------------------------------------
% ..........main iterative loop..........
%---------------------------------------------------------
% compute search direction
%---------------------------------------------------------
argvec = [accrcy gnorm xnorm];
[p, gtp, ncg1, d, eig_val,condnum] = ...
     modlnp (d, x, g, maxit, upd1, ireset, 0, ipivot, argvec, sfun);
ncg = ncg + ncg1;

infos.eigen_val = eig_val;
infos.condNum = condnum;

mse_values = [];
mse_iter_imag = [];
mse_iter_real = [];

nmse_values = [];

loss_ptyc = [];
loss_fluor = [];

RMSE_Phase = [];
SSIM_magnitude = [];

SSIM_phase = [];
SSIM_fluor = [];

Ne = 1;
 
while (~conv);
    oldg   = g;
    pnorm  = norm(p,'inf');
    oldf   = f;
    %---------------------------------------------------------
    % line search
    %---------------------------------------------------------
    pe     = pnorm + eps;
    spe    = stepmx/pe;
    alpha0 = step1 (f, gtp, spe);

    
    % mse_iter = mean(abs(obj - obj_true).^2, 'all'); 

    % norm_diff = norm(abs(x - v_o_true), 'all'); 


    mse_iter = mean(abs(x- v_o_true).^2, 'all');
    mse_values = [mse_values; mse_iter]; % Ensure correct dimensions

    err = x - v_o_true;    
    nmse_iter = sum(abs(err).^2, 'all') / max(sum(abs(v_o_true).^2, 'all'), eps);
    nmse_values = [nmse_values; nmse_iter];  

        


    if separate_loss == 1

        mse_iter_imag_value = mean( abs(x(N_obj^2*Ne+1 : end) - v_o_true(N_obj^2*Ne + 1:end)).^2 , 'all');
        mse_iter_imag = [mse_iter_imag; mse_iter_imag_value];

        mse_iter_real_value = mean(abs(x (1: N_obj^2) - v_o_true(1:N_obj^2)).^2, 'all');
        mse_iter_real = [mse_iter_real; mse_iter_real_value]; 

        object = realToComplex(x);        
        object_true = realToComplex(v_o_true);


        % mse_iter_imag_value = mean(abs( -log(abs(object(:))) - w_true(:)).^2, 'all');
        % mse_iter_imag = [mse_iter_imag; mse_iter_imag_value];



        % SSIM metric
        % 
        % SSIM_mag_step = ssim(abs(object), abs(object_true));
        % SSIM_magnitude = [SSIM_magnitude; SSIM_mag_step];
        % 
        % 
        % SSIM_phase_step = ssim(angle(object), angle(object_true));
        % SSIM_phase = [SSIM_phase; SSIM_phase_step];
        % 
        % 
        % % SSIM_fluor_step = ssim(-log(abs(object)), w_true);
        % 
        % Imaginary_plot = x(N_obj^2+1: end);
        % 
        % SSIM_fluor_step = ssim(reshape(Imaginary_plot, N_obj, N_obj), w_true);
        % SSIM_fluor = [SSIM_fluor; SSIM_fluor_step];
        % 


        loss_ptyc_step = fctn_o(x);
        loss_ptyc = [loss_ptyc; loss_ptyc_step];

        loss_fluor_step = fctn_f(x(N_obj^2 * Ne+1:end));
        loss_fluor = [loss_fluor; loss_fluor_step];


    end

    [x, f, g, nf1, ierror, alpha] = lin1 (p, x, f, alpha0, g, sfun);

    



    % figure(10), plot(p);%imagesc(abs(realToComplex(x)))
    % pause(1)
    % drawnow;
    %%%=============================================================
    nf = nf + nf1;
    %---------------------------------------------------------
    nit = nit + 1;
    gnorm = norm(g,'inf');
    %%==============store iterative info
    infos.epoch = [infos.epoch nit];
    infos.grad_calc_count = [infos.grad_calc_count nf+ncg];      
    infos.cost = [infos.cost f];
    infos.gnorm = [infos.gnorm gnorm];
    infos.mse_values = mse_values; % Store MSE in `infos`

    infos.mse_values_imag = mse_iter_imag;
    infos.mse_values_real = mse_iter_real;

    infos.cost_ptyc = loss_ptyc;
    infos.cost_fluor = loss_fluor;

    infos.RMSE = RMSE_Phase;
    infos.SSIM_magnitude = SSIM_magnitude;

    infos.SSIM_phase = SSIM_phase;

    infos.nmse_values = nmse_values;
    infos.fluor_ssim = SSIM_fluor;


    %%=================================
    gv=g;
    ptest=p;
    fprintf(1,'%4i   %4i   %4i   % .8e   %.1e\n', ...
        nit, nf, ncg, f, gnorm)
    if (ierror == 3);
        if isempty(ncg); ncg = 0; end;
        xstar = x;
        % NF(1,nind) = NF(1,nind) + nit;
        % NF(2,nind) = NF(2,nind) + nf;
        % NF(3,nind) = NF(3,nind) + ncg;
        disp('ierror=3');
        return;
    end;
    %---------------------------------------------------------
    % stop if more than maxfun evalutations have been made
    %---------------------------------------------------------
    if (nf >= maxfun);
        ierror = 2;
        xstar = x;
        NF(1,nind) = NF(1,nind) + nit;
        NF(2,nind) = NF(2,nind) + nf;
        NF(3,nind) = NF(3,nind) + ncg;
        return;
    end;
    %---------------------------------------------------------
    % set up for convergence and resetting tests
    %---------------------------------------------------------
    ftest  = 1 + abs(f);
    xnorm  = norm(x,'inf');
    difold = difnew;
    difnew = oldf - f;
    yk     = g - oldg;
    sk     = alpha*p;
    if (icycle == 1);
        if (difnew >   2*difold); epsred =   2*epsred; end;
        if (difnew < 0.5*difold); epsred = 0.5*epsred; end;
    end;
    %---------------------------------------------------------
    % convergence test
    %---------------------------------------------------------
    conv = (alpha*pnorm < toleps*(1 + xnorm) ... % change the converge criteria
        & abs(difnew) < rtleps*ftest  ...
        & gnorm < accrcy^(1/3)*ftest)    ...
        | gnorm < .01*sqrt(accrcy)*ftest; % change from 0.01 to 0.0001

    % disp( .01*sqrt(accrcy)*ftest);
    % conv = gnorm < 1e-5; %%Wendy's simplied stoppping criteria
    %+++++++++++++++++++++++++++++++++++++++++++++++++++
    % conv = (gnorm < 1d-6*ftest);
    %+++++++++++++++++++++++++++++++++++++++++++++++++++
    if (conv | nit>=maxiter);
        ierror = 0;
        xstar = x;
        % NF(1,nind) = NF(1,nind) + nit;
        % NF(2,nind) = NF(2,nind) + nf;
        % NF(3,nind) = NF(3,nind) + ncg;

        if eig_val_need == 1


            H = approximate_hessian(x, g, accrcy, xnorm, sfun);
            H_final = (H + H')/2;
            eig_values = eig(H_final);
            infos.eigenvalues = eig_values;
            % conditional_number = cond(H_final);
            % % 
            % disp("Condition number")
            % disp(conditional_number); 
        end
      

        return;
    end;
    %---------------------------------------------------------
    % update lmqn preconditioner
    %---------------------------------------------------------
    yksk = yk'*sk;
    ireset = (icycle == n-1 | difnew < epsred*(fkeep-f));
    if (~ireset);
        yrsr = yr'*sr;
        ireset = (yrsr <= 0);
    end;
    upd1 = (yksk <= 0);
    %---------------------------------------------------------
    % compute search direction
    %---------------------------------------------------------
    argvec = [accrcy gnorm xnorm];



    [p, gtp, ncg1, d, eig_val,condnum] = ...
        modlnp (d, x, g, maxit, upd1, ireset, 0, ipivot, argvec, sfun);
    ncg = ncg + ncg1;


    infos.eigen_val = eig_val;
    infos.condNum = condnum;
    %---------------------------------------------------------
    % store information for lmqn preconditioner
    %---------------------------------------------------------
    if (ireset);
        sr = sk;
        yr = yk;
        fkeep = f;
        icycle = 1;
    else
        sr = sr + sk;
        yr = yr + yk;
        icycle = icycle + 1;
    end;
end;
