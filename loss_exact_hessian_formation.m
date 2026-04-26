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

%% distribution plot

[max_lambda_p, imax_p] = max(lambda_p);    
[min_lambda_p, imin_p] = min(lambda_p);
[max_lambda_j, imax_j] = max(lambda_j);    
[min_lambda_j, imin_j] = min(lambda_j);


lambda_p = lambda_p/max_lambda_p;
lambda_j = lambda_j/max_lambda_j;

[lambda_p, idx_p] = sort(lambda_p, 'descend');
[lambda_j, idx_j] = sort(lambda_j, 'descend');

thr_p = -1;                       
thr_j = -1;                      

posIdx_p = find(lambda_p > thr_p);          
posIdx_j = find(lambda_j > thr_j);          

nPos_p   = numel(posIdx_p);
nPos_j   = numel(posIdx_j);

if nPos_p < 2 || nPos_j < 2
    error('Not enough positive λ to take the requested percentiles.');
end

    max_joint_eig = max(lambda_j);
    max_ptyo_eig = max(lambda_p);
    
    upper_percentile = 100;
    lower_joint = prctile(lambda_j, 0);
    upper_joint = prctile(lambda_j, upper_percentile);
    eig_joint_zoomed = lambda_j(lambda_j >= lower_joint & lambda_j <= upper_joint);
    eig_joint_zoomed = eig_joint_zoomed / max_joint_eig;  
    
    lower_ptyo = prctile(lambda_p, 0);
    upper_ptyo = prctile(lambda_p, upper_percentile);
    eig_ptyo_zoomed = lambda_p(lambda_p >= lower_ptyo & lambda_p <= upper_ptyo);
    eig_ptyo_zoomed = eig_ptyo_zoomed / max_ptyo_eig;  
    
    upperbound = 1;
    bin_edges  = linspace(0, upperbound, 21);
    
    [counts_joint, ~] = histcounts(eig_joint_zoomed, bin_edges);
    [counts_ptyo,  ~] = histcounts(eig_ptyo_zoomed,  bin_edges);
    max_count = max([counts_joint, counts_ptyo]);
    min_count = 1;
    
    figure('Color','w');  % white background
    
    ax1 = subplot(1,2,1);
    histogram(eig_joint_zoomed, 'BinEdges', bin_edges, ...
        'FaceColor', [0.4 0.2 0.8], 'EdgeColor', 'none', 'FaceAlpha', 0.3);
    xlabel('Normalized Eigenvalue', 'FontSize', 18);
    ylabel('Count',                 'FontSize', 18);
    title(sprintf('Joint Method (0–%dth Percentile Eigvals)', upper_percentile), 'FontSize', 25);
    xlim([0, upperbound]); ylim([min_count, max_count]);
    set(ax1, 'YScale', 'log', 'LineWidth', 1.2, 'FontSize', 20);
    grid on; box off;
    
    ax2 = subplot(1,2,2);
    histogram(eig_ptyo_zoomed, 'BinEdges', bin_edges, ...
        'FaceColor', [0.2 0.7 0.4], 'EdgeColor', 'none', 'FaceAlpha', 0.5);
    xlabel('Normalized Eigenvalue', 'FontSize', 18);
    ylabel('Count',                 'FontSize', 18);
    title(sprintf('Ptychography Method (0–%dth Percentile Eigvals)', upper_percentile), 'FontSize', 25);
    xlim([0, upperbound]); ylim([min_count, max_count]);
    set(ax2, 'YScale', 'log', 'LineWidth', 1.2, 'FontSize', 20);
    grid on; box off;

%% loss surface




d1_ptyo   = Vp(:,2000);   %3e-4
d1_ptyo   = d1_ptyo / norm(d1_ptyo);
d2_ptyo   = Vp(:, 2001);   
d2_ptyo   = d2_ptyo / norm(d2_ptyo);

d1_joint  = Vj(:, 2000); % 3e-4
d1_joint  = d1_joint / norm(d1_joint);
d2_joint  = Vj(:, 2001);   
d2_joint  = d2_joint / norm(d2_joint); %1e-4

range_a   = 5e-14;    % perturbation size along eigenvector directions % 3e-14
range_b   = 5e-14;  % perturbation size along smallest‐λ direction

Ngrid     = 125;
alphas    = linspace(-range_a, range_a, Ngrid);
betas     = linspace(-range_b, range_b, Ngrid);
[AA, BB]  = meshgrid(alphas, betas);

% --- 2) Evaluate loss on the (α,β) grid ---
Z_ptyo    = zeros(Ngrid, Ngrid);
Z_joint   = zeros(Ngrid, Ngrid);
for i = 1:Ngrid
  for j = 1:Ngrid
    theta_p         = v_o_true  + AA(i,j)*d1_ptyo  + BB(i,j)*d2_ptyo;
    theta_j         = v_o_true + AA(i,j)*d1_joint + BB(i,j)*d2_joint;
    Z_ptyo(i,j)     = fctn_o(theta_p);
    Z_joint(i,j)    = fctn_j(theta_j);
  end
end

%%

expA = floor(log10(range_a));
facA = 10^expA;
expB = floor(log10(range_b));
facB = 10^expB;

As = AA / facA;   % now runs roughly -5..5
Bs = BB / facB;

%%

[~, i0] = min(abs(betas));   % β index  (row)
[~, j0] = min(abs(alphas));  % α index  (col)

alpha0 = 0;      beta0  = 0;
z0_pty = Z_ptyo (i0,j0);
z0_jnt = Z_joint(i0,j0);

%%

expP         = floor(log10(max(Z_ptyo(:))));
expJ         = floor(log10(max(Z_joint(:))));


commonExp    = -1;
scaleP       = 10^(commonExp - expP);
scaleJ       = 10^(commonExp - expJ);


Z_ptyo = Z_ptyo * scaleP;     
Z_joint = Z_joint * scaleJ;   



%% 

figure('Color','w','Position',[50 50 1600 1100]);
t = tiledlayout(2,2,'TileSpacing','compact','Padding','compact');

%% Top‐left: Ptychography surface
ax1 = nexttile;
surf(ax1, As, Bs, Z_ptyo, 'EdgeColor','none','FaceAlpha',0.85); hold(ax1,'on');
mesh(ax1, As, Bs, Z_ptyo, 'EdgeColor',[.3 .3 .3],'FaceColor','none');
view(ax1,30,45), camlight(ax1,'headlight'), lighting(ax1,'phong');
axis(ax1,'tight'); grid(ax1,'on');
xlabel(ax1,'\alpha'), ylabel(ax1,'\beta'), zlabel(ax1,'Loss_{ptyo}');
title(ax1,'Ptychography – Loss surface','FontWeight','bold');
hold(ax1,'on');
plot3(ax1, alpha0, beta0, z0_pty, 'ro', 'MarkerSize',8,'MarkerFaceColor','r');

%% Top‐right: Joint surface
ax2 = nexttile;
surf(ax2, As, Bs, Z_joint, 'EdgeColor','none','FaceAlpha',0.85); hold(ax2,'on');
mesh(ax2, As, Bs, Z_joint, 'EdgeColor',[.3 .3 .3],'FaceColor','none');
view(ax2,30,45), camlight(ax2,'headlight'), lighting(ax2,'phong');
axis(ax2,'tight'); grid(ax2,'on');
xlabel(ax2,'\alpha'), ylabel(ax2,'\beta'), zlabel(ax2,'Loss_{joint}');
title(ax2,'Joint – Loss surface','FontWeight','bold');
hold(ax2,'on');
plot3(ax2, alpha0, beta0, z0_jnt, 'ro', 'MarkerSize',8,'MarkerFaceColor','r');

%% Bottom‐left: Ptychography contour
ax3 = nexttile(3);
contourf(ax3, As, Bs, Z_ptyo, 30, 'LineColor','none');
axis(ax3,'equal','tight');
xlabel(ax3,'\alpha'); ylabel(ax3,'\beta');
title(ax3,'Ptychography – Contour','FontWeight','bold');

%% Bottom‐right: Joint contour
ax4 = nexttile(4);
contourf(ax4, As, Bs, Z_joint, 30, 'LineColor','none');
axis(ax4,'equal','tight');
xlabel(ax4,'\alpha'); ylabel(ax4,'\beta');
title(ax4,'Joint – Contour','FontWeight','bold');

%% ------------------------------------------------
% Shared colorbars: one per column
axH = [ax1 ax2; ax3 ax4];  % arrange in 2x2 for indexing
cbW  = 0.01;   % thin colorbar
gap  = 0.01;   % gap between column and bar

for col = 1:2
    % top and bottom axes in this column
    posTop    = get(axH(1,col), 'Position');
    posBottom = get(axH(2,col), 'Position');

    % compute span of both rows
    yBottom = posBottom(2);
    yTop    = posTop(2) + posTop(4);
    height  = yTop - yBottom;
    xRight  = posTop(1) + posTop(3) + gap;

    % attach colorbar to the TOP axes in this column
    cb = colorbar(axH(1,col), 'eastoutside');
    cb.Position = [xRight, yBottom, cbW, height];

    % label + scaling + rainbow colormap
    if col==1
        caxis(ax1,[min(Z_ptyo(:)) max(Z_ptyo(:))]);
        caxis(ax3,[min(Z_ptyo(:)) max(Z_ptyo(:))]);
        colormap(ax1,jet); colormap(ax3,jet);
    else
        caxis(ax2,[min(Z_joint(:)) max(Z_joint(:))]);
        caxis(ax4,[min(Z_joint(:)) max(Z_joint(:))]);
        colormap(ax2,jet); colormap(ax4,jet);
    end
end