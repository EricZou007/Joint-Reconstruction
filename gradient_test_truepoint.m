global NF fiter maxiter itertest
global N Np Ne N_obj
global H_init
global beta1 thres

N_probe = 16;  % 16, 32, 66, 1e-7
N_scan = 16;
N_obj_size = 62;
stepSize = 1.5e-7;
Nc_avg = inf;
eig_val_need = 0; 

true_value = 0;
beta_1 = 0.001 ; 


beta1 = beta_1;
stepSize_true = stepSize;
Ne = 1;


load(['data_Eric/Np', num2str(N_probe), '_multimodal_', num2str(Ne), '_', ...
      num2str(stepSize), '_', num2str(N_scan), '_N_obj', num2str(N_obj_size),'_noise',num2str(Nc_avg), '_aperiodic.mat']);

Zr_true = Zr;
O_true = O;

v_Zr_true = Zr_true(:);
v_O_true = O_true(:); 


v_o_true =[Zr(:); O(:)];

H_init = 'standard';
dia_pert = 2;
do_setup;
thres = 1e-10;  % Threshold for postivity
N = N_obj^2 * (Ne + 1);
NF = [0 * N; 0 * N; 0 * N];

v_obj_true = [Zr(:); O(:)];
probe = probe_true; % Fix probe at truth
v_p = complexToReal(probe);
O = ones(N_obj, N_obj, Ne);
Zr = ones(N_obj, N_obj);
v_o = [Zr(:); O(:)];
v_o_init = v_o; 
indices = 1:N_scan;


fctn_o = @(o) sfun_o(o, p, dp);
fctn_f= @(o) func_conv_o(o,p,d);
[~, grad_ptyo] = sfun_o(v_o_init, probe_true, dp);
[~, grad_fluor] = func_conv_o(v_o_init(N_obj^2+1:end), probe_true ,d);
% beta1 = sqrt((norm(grad_ptyo)))/sqrt((norm(grad_fluor)));
beta1 = ((norm(grad_ptyo)))/((norm(grad_fluor)))  ;
disp(beta1);

beta1 = 0.001;

%% Hessian Formation Prcoess

z = obj(:);

probe_abs = abs(probe_true);

[Jf,~] = convolutionMatrix(probe_abs.^2,N_obj,Np);

resF = Jf*real(z) - d(:);

F=fftshift(fft(ifftshift(eye(Np))));
F=kron(F,F);

F_inv = fftshift(ifft(ifftshift(eye(Np))));
F_inv = kron(F_inv,F_inv);

Jp = [];
P = [];
res = [];
res2 = [];

for i=1:N_scan

    [ind_X,ind_Y]=meshgrid(ind_b(i,1):ind_b(i,2),ind_b(i,3):ind_b(i,4));
    vecInd=sub2ind([N_obj,N_obj],ind_Y(:),ind_X(:));
    Pj=sparse(Np^2,N_obj^2);
    for k=1:Np^2
        Pj(k,vecInd(k))=probe(k); %P_j
    end

    dp_j = dp(:,:,i);
    
    % data misfit
    psi_old = Pj*z;
    psi = F*(Pj*z);

  

    absp = abs(psi(:))+0e-6;
    res = [res ; absp - dp_j(:)];

   
    J_j = conj(diag(psi(:)./absp))*(F*Pj);


    Jp = [Jp;J_j];

    % feasibility
    psi2 = (psi./absp).*dp_j(:);
    psi2 = F_inv*psi2;
    res2 = [res2; psi_old - psi2];

    P = [P;Pj];
end

J = [Jf;Jp];

[f_eval,grad_eval] = sfun_o(v_obj_true,probe_true, dp);
[f_eval_1,grad_evalF] = func_conv_o(v_obj_true(N_obj^2+1:end),probe_true,d);

[f_eval_2, grad_evalJ] = sfun_joint(v_obj_true, probe_true, d, dp);


Hf = Jf'*Jf;

HF = full(Hf);



%%
[f_ptyo,g_ptyo,H_ptyo] = sfun_o_pie(v_o_true, probe, dp, ind_b, N_obj, Np, indices);

idx = N_obj^2 + (1:N_obj^2);


H1 = H_ptyo;              
H1(idx, idx)  = H1(idx, idx) + beta_1 * Hf;

H_joint = H1;


%% 
fctn_j = @(o) sfun_joint(o, probe, d, dp);
fctn_o = @(o) sfun_o(o, probe, dp);

[Vp,Dp] = eig(full(H_ptyo));
lambda_p = diag(Dp);

% Sort in descending order
[lambda_p, idx_p] = sort(lambda_p, 'descend');
Vp = Vp(:, idx_p);
Dp = diag(lambda_p);

% For H_joint
[Vj,Dj] = eig(full(H_joint));
lambda_j = diag(Dj);

% Sort in descending order
[lambda_j, idx_j] = sort(lambda_j, 'descend');
Vj = Vj(:, idx_j);
Dj = diag(lambda_j);




d1_ptyo   = Vp(:,1);   %3e-4
d1_ptyo   = d1_ptyo / norm(d1_ptyo);


d1_joint  = Vj(:, 1); % 3e-4
d1_joint  = d1_joint / norm(d1_joint);




step = 1e-1;            % given
N = 601;
N_center = 501;          % keep odd to include 0

W_phys   = 5e-14;        % center window in *physical* units
eps_band = min(1, W_phys/step);   % convert to epsilon units

N_wing = max(0, N - N_center);

left  = linspace(-1,        -eps_band, floor(N_wing/2));
mid   = linspace(-eps_band, +eps_band, N_center);
right = linspace(+eps_band, +1,        ceil(N_wing/2));

epsilons = unique([left, mid, right], 'stable');

gnorm_joint = zeros(size(epsilons));
gnorm_ptyo  = zeros(size(epsilons));
gnorm_fluor = zeros(size(epsilons));

loss_joint = zeros(size(epsilons));
loss_ptyc  = zeros(size(epsilons));
loss_fluor = zeros(size(epsilons));

for k = 1:numel(epsilons)
    v_pert_joint = v_o_true;
    v_pert_ptyc  = v_o_true;
    v_pert_fluor = v_o_true(N_obj^2+1:end);

    % Joint perturbation
    v_pert_joint(:) = v_pert_joint(:) + (step * epsilons(k)) * d1_joint;
    [f_j, g_j] = sfun_joint(v_pert_joint, probe, d, dp);
    gnorm_joint(k) = norm(g_j, 2);
    loss_joint(k) = f_j;

    % Ptychography perturbation
    v_pert_ptyc(:) = v_pert_ptyc(:) + (step * epsilons(k)) * d1_ptyo;
    [f_p, g_p] = sfun_o(v_pert_ptyc, probe, dp);
    gnorm_ptyo(k) = norm(g_p, 2);
    loss_ptyc(k) = f_p;

    % Fluorescence perturbation
    v_pert_fluor(:) = v_pert_fluor(:) + (step * epsilons(k)) * d1_joint(N_obj^2+1:end);
    [f_f, g_f] = func_conv_o(v_pert_fluor, probe, d);
    gnorm_fluor(k) = norm(g_f, 2);





    loss_fluor(k) = f_f;
end

%%



x_phys = step * epsilons;
x0 = max(min(abs(x_phys(x_phys~=0))), 1e-20);
sx = @(x) sign(x).*log10(1 + abs(x)/x0);
xplot = sx(x_phys);

figure; hold on;

hJ = plot(xplot, gnorm_joint, '-o', 'LineWidth', 3, ...
          'DisplayName', 'Joint');
hP = plot(xplot, gnorm_ptyo, '-x', 'LineWidth', 3, ...
          'DisplayName', 'Ptychography');
hF = plot(xplot, gnorm_fluor, '-s', 'LineWidth', 3, ...
          'DisplayName', 'Fluorescence');

set(gca, 'YScale', 'log');
grid on;

ticks_phys = [-1e-2 -1e-4 -1e-6 -1e-8 -1e-10 -1e-12 -1e-14 0 ...
               1e-14  1e-12  1e-10  1e-8  1e-6  1e-4  1e-2];
ticks_phys = ticks_phys(abs(ticks_phys) <= max(abs(x_phys)));
set(gca, 'XTick', sx(ticks_phys), 'XTickLabel', compose('%.0e', ticks_phys));

xlabel('Perturbation size log-scale', 'FontSize', 22);
ylabel('2-norm of gradient', 'FontSize', 22);
title('Gradient Norm Perturbation around True Point', 'FontSize', 22);

lgd = legend([hJ hP hF], {'Joint', 'Ptychography', 'Fluorescence'}, ...
             'Location', 'north', 'FontSize', 20);

% -----------------------
% Annotation points
% -----------------------
idxJ = min(80, numel(epsilons));
idxP = min(40, numel(epsilons));
idxF = min(60, numel(epsilons));

xJ = xplot(idxJ); yJ = gnorm_joint(idxJ);
xP = xplot(idxP); yP = gnorm_ptyo(idxP);
xF = xplot(idxF); yF = gnorm_fluor(idxF);

ax = gca;
xp = ax.XLim;
yp = ax.YLim;

dx = 0.04 * (xp(2)-xp(1));
y_dec_span = log10(yp(2)) - log10(max(yp(1), realmin));
up_factor   = 10^(+0.30 * y_dec_span);
down_factor = 10^(-0.05 * y_dec_span);

% Joint label
xJ_lab = xJ + 1.4;
yJ_lab = yJ * up_factor;
text(xJ_lab, yJ_lab, 'Joint gradient', ...
     'Color', 'b', 'FontSize', 20, ...
     'HorizontalAlignment', 'center', ...
     'VerticalAlignment', 'bottom', ...
     'HandleVisibility', 'off');
quiver(xJ_lab, yJ_lab, xJ - xJ_lab, yJ - yJ_lab, 0, ...
       'Color', 'b', 'MaxHeadSize', 0.0005, ...
       'LineWidth', 1.2, 'HandleVisibility', 'off');

% Ptychography label
xP_lab = xP;
yP_lab = yP * down_factor;
text(xP_lab, yP_lab, 'Ptychography gradient', ...
     'Color', 'r', 'FontSize', 20, ...
     'HorizontalAlignment', 'center', ...
     'VerticalAlignment', 'top', ...
     'HandleVisibility', 'off');
quiver(xP_lab, yP_lab, xP - xP_lab, yP - yP_lab, 0, ...
       'Color', 'r', 'MaxHeadSize', 0.0005, ...
       'LineWidth', 1.2, 'HandleVisibility', 'off');

% Fluorescence label
xF_lab = xF - 1.2;
yF_lab = yF * up_factor;
text(xF_lab, yF_lab, 'Fluorescence gradient', ...
     'Color', [0 0.5 0], 'FontSize', 20, ...
     'HorizontalAlignment', 'center', ...
     'VerticalAlignment', 'bottom', ...
     'HandleVisibility', 'off');
quiver(xF_lab, yF_lab, xF - xF_lab, yF - yF_lab, 0, ...
       'Color', [0 0.5 0], 'MaxHeadSize', 0.0005, ...
       'LineWidth', 1.2, 'HandleVisibility', 'off');

hold off;



%%

x_phys = step * epsilons;
x0 = max(min(abs(x_phys(x_phys~=0))), 1e-20);   
sx = @(x) sign(x).*log10(1 + abs(x)/x0);     
xplot = sx(x_phys);



figure; hold on;


hJ = plot(xplot, gnorm_joint, '-o', 'LineWidth', 3, ...
          'DisplayName','Joint');
hP = plot(xplot, gnorm_ptyo,  '-x', 'LineWidth', 3, ...
          'DisplayName','Ptychography');

set(gca,'YScale','log'); 
grid on;

% Symmetric-log ticks (as you had)
ticks_phys = [-1e-2 -1e-4 -1e-6 -1e-8 -1e-10 -1e-12 -1e-14 0 1e-14 1e-12 1e-10 1e-8 1e-6 1e-4 1e-2];
ticks_phys = ticks_phys(abs(ticks_phys) <= max(abs(x_phys)));
set(gca,'XTick', sx(ticks_phys), 'XTickLabel', compose('%.0e', ticks_phys));

xlabel('Perturbation size log-scale', 'FontSize', 22);
ylabel('2-norm of gradient',    'FontSize', 22);
title('Gradient Norm Perturbation around True Point', 'FontSize', 22);

% --- Legend ---
lgd = legend([hJ hP], {'Joint','Ptychography'}, ...
             'Location','north', 'FontSize', 20);

idxJ = min(80, numel(epsilons));   % Joint index (clamped)
idxP = min(40, numel(epsilons));   % Ptychography index (clamped)

% points in transformed-x space (symmetric-log) and log-y axis
xJ = xplot(idxJ);   yJ = gnorm_joint(idxJ);
xP = xplot(idxP);   yP = gnorm_ptyo(idxP);

% Compute data-range-based offsets
ax = gca;
xp = ax.XLim;     yp = ax.YLim;          % x in transformed coords; y is log-scale
dx = 0.04 * (xp(2)-xp(1));               % 4% of x-range
y_dec_span = log10(yp(2)) - log10(max(yp(1), realmin));
up_factor   = 10^(+0.30 * y_dec_span);   % ~10% of vertical span up
down_factor = 10^(-0.05 * y_dec_span);   % ~8%  of vertical span down

% Joint label (above point)
xJ_lab = xJ + 1.4;                 % set to xJ ± dx if you want a horizontal nudge
yJ_lab = yJ * up_factor;
text(xJ_lab, yJ_lab, 'Converge point (high loss, low MSE)', ...
     'Color','b','FontSize',22,'HorizontalAlignment','center', ...
     'VerticalAlignment','bottom','HandleVisibility','off');
quiver(xJ_lab, yJ_lab, xJ - xJ_lab, yJ - yJ_lab, 0, ...
       'Color','b','MaxHeadSize',0.0005,'LineWidth',1.2,'HandleVisibility','off');

% Ptychography label (below point)
xP_lab = xP;                 % set to xP ± dx if you want a horizontal nudge
yP_lab = yP * down_factor;
text(xP_lab, yP_lab, 'Converge point (low loss, high MSE)', ...
     'Color','r','FontSize',22,'HorizontalAlignment','center', ...
     'VerticalAlignment','top','HandleVisibility','off');
quiver(xP_lab, yP_lab, xP - xP_lab, yP - yP_lab, 0, ...
       'Color','r','MaxHeadSize',0.0005,'LineWidth',1.2,'HandleVisibility','off');

hold off;