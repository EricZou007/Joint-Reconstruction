global NF fiter maxiter itertest
global N Np N_obj
global H_init 
global beta1 thres

beta1 = 0.0001;
N_probe = 64; 
N_scan = 100;
N_obj_size = 334;
stepSize = 3e-7;
Nc_avg = inf;
eig_val_need = 0; 
true_value = 0;
stepSize_true = stepSize;
Ne = 3;


load(['data_Eric/Np', num2str(N_probe), '_multimodal_', num2str(Ne), '_', ...
      num2str(stepSize), '_', num2str(N_scan), '_N_obj', num2str(N_obj_size),'_noise',num2str(Nc_avg), '_aperiodic.mat']);


Zr_true = Zr;
O_true = O;

v_O_true = O_true(:); 
v_Zr_true = Zr_true(:);
v_o_true =[Zr(:); O(:)];

H_init = 'standard';
dia_pert = 2;
do_setup;
thres = 1e-10;  
N = N_obj^2 * (Ne + 1);
NF = [0 * N; 0 * N; 0 * N];

v_obj_true = [Zr(:); O(:)];
p = probe_true;
v_p = complexToReal(p);
O = ones(N_obj, N_obj, Ne);
Zr = ones(N_obj, N_obj);


v_o_joint = [Zr(:); O(:)];
v_o_ptyc = [Zr(:); O(:)];


separate_loss = 0;
    

%%

f_tot_joint = [];
mse_joint = [];
gradient_joint = [];
f_tot_joint_ptyc = [];

if true_value == 1
    v_o = v_o_true;
end

separate_loss = 0;
outermax = 1;

maxiter = 100;

fctn_o = @(o) sfun_o_multiple_ne(o, probe_true, dp, Ne);
fctn_f= @(o) func_conv_o(o,p,d);



 

    %% Joint Reconstruction Loop
    for k = 1:outermax    
        disp(['============ Joint Reconstruction, Iteration: ', num2str(k)])
        
        fctn_j = @(o) sfun_joint_multiple_ne(o, p, d, dp, Ne);  
        [v_o_joint, f, g] = tn(v_o_joint, fctn_j, v_o_true, eig_val_need, separate_loss);

        
        f_tot = f.cost(); 
        mse_joint = [mse_joint; f.mse_values];
        gradient_joint = [gradient_joint; f.gnorm()'];

        
        Zr_joint = reshape(v_o_joint(1: N_obj^2), N_obj, N_obj); 
        O_joint = reshape(v_o_joint(N_obj^2+1: end), N_obj, N_obj,Ne);
        v_o_joint = [Zr_joint(:); O_joint(:)];
      
        f_tot_joint = [f_tot_joint; f_tot'];
        f_tot_joint_ptyc = [f_tot_joint_ptyc; f.cost_ptyc];

    end

    %% ptychography

    f_tot_ptyc = [];
    mse_ptyc = []; 
    gradient_ptyo = [];
    separate_loss = 0;



    for k = 1:outermax
        disp(['============ Outer iteration ', num2str(k), ' / ', num2str(outermax), ' ============']);
        

        fctn_o = @(o) sfun_o_multiple_ne(o, probe_true, dp, Ne);
       
        [v_o_ptyc, f, g] = tn(v_o_ptyc, fctn_o, v_o_true, eig_val_need, separate_loss);

        f_tot = f.cost(); 
        gradient_ptyo_tot = f.gnorm();
        mse_ptyc = [mse_ptyc ;f.mse_values];

        
        Zr_ptyc = reshape(v_o_ptyc(1: N_obj^2), N_obj, N_obj); 
        O_ptyc = reshape(v_o_ptyc(N_obj^2+1: end), N_obj, N_obj,Ne);

        v_o_ptyc = [Zr_ptyc(:); O_ptyc(:)]; 
      
        f_tot_ptyc = [f_tot_ptyc; f_tot'];
        gradient_ptyo = [gradient_ptyo; gradient_ptyo_tot'];
    end

%% plot

%% --- FIX: prepare selectors (no literal cell indexing) ---

Zr_set = {Zr_true, Zr_ptyc, Zr_joint};
O_set  = {O_true,  O_ptyc,  O_joint};


getZr = @(r) Zr_set{r};
getO  = @(r) O_set{r};

Ne_all  = size(O_true, 3);
Ne_show = min(3, Ne_all);

colNames = cell(1, 1+Ne_show);
colNames{1} = 'Real Part';
for k = 1:Ne_show
    if exist('energies','var') && numel(energies) >= k
        colNames{1+k} = sprintf('Imaginary (ch %d: %.3g)', k, energies(k));
    else
        colNames{1+k} = sprintf('Imaginary Part (channel %d)', k);
    end
end
rowNames = {'True','Ptychography','Joint (ours)'};

% How far left (in axes-normalized units) to place the label
% labelOffset = 0.1;   % increase if you want it further left

%% --- Shared limits ---
real_min = min([Zr_true(:); Zr_ptyc(:); Zr_joint(:)]);
real_max = max([Zr_true(:); Zr_ptyc(:); Zr_joint(:)]);

imag_min = zeros(Ne_show,1);
imag_max = zeros(Ne_show,1);
for k = 1:Ne_show
    vals_all = cat(3, O_true(:,:,k), O_ptyc(:,:,k), O_joint(:,:,k)); 
    imag_min(k) = min(vals_all, [], 'all');
    imag_max(k) = max(vals_all, [], 'all');
end

%% --- Figure & layout ---
fig = figure('Units','pixels','Position',[100 100 1650 1050]);
tlo = tiledlayout(3, 1+Ne_show, 'TileSpacing','compact','Padding','compact');
colormap(fig, parula);

% Set global defaults for readability
set(groot,'DefaultAxesFontSize',12);
set(groot,'DefaultColorbarFontSize',12);

ax = gobjects(3, 1+Ne_show);

for r = 1:3
    % Column 1: Real (Zr)
    ax(r,1) = nexttile((r-1)*(1+Ne_show) + 1);
    imagesc(getZr(r));
    axis image;
    set(ax(r,1),'XTick',[],'YTick',[], 'Box','off','TickLength',[0 0]);
    caxis(ax(r,1), [real_min real_max]);
    if r == 1
        title(colNames{1}, 'FontWeight','bold','FontSize',20);
    end


     text(-0.1, 0.5, rowNames{r}, ...
        'Units','normalized', ...   % relative to axis
    'FontWeight','bold', ...
    'FontSize',25, ...
    'HorizontalAlignment','center', ...
    'VerticalAlignment','middle', ...
    'Rotation',90);

    % Columns 2..(1+Ne_show): Imaginary channels O(:,:,k)
    Ocur = getO(r);
    for k = 1:Ne_show
        ax(r,1+k) = nexttile((r-1)*(1+Ne_show) + 1 + k);
        imagesc(Ocur(:,:,k));
        axis image off;
        caxis(ax(r,1+k), [imag_min(k) imag_max(k)]);
        if r == 1
            title(colNames{1+k}, 'FontWeight','bold','FontSize',20);
        end
    end
end

%% --- One tall colorbar per column (aligned) ---
cbWidth = 0.015;  % normalized figure units
xGap    = 0.010;

for c = 1:(1+Ne_show)
    posTop    = get(ax(1,c), 'Position');
    posBottom = get(ax(3,c), 'Position');

    yBottom = posBottom(2);
    yTop    = posTop(2) + posTop(4);
    height  = yTop - yBottom;
    xRight  = posTop(1) + posTop(3) + xGap;

    cb = colorbar(ax(1,c), 'Location','eastoutside');
    cb.Position = [xRight, yBottom, cbWidth, height];
    cb.FontSize = 14; % make ticks readable
end

%%

num_iterations_error = maxiter * outermax;
num_iterations_loss = maxiter * outermax;

% 2) ERROR (MSE) PLOT
figure('Units','inches','Position',[1 1 5 5], ...            % same square size
       'PaperUnits','inches','PaperPosition',[0 0 5 5]);
hJ2 = loglog(1:length(mse_joint), mse_joint,'-o', ...
             'LineWidth',1.5,'MarkerSize',4,'Color',[0 0 1]); % Blue
hold on;
hP2 = loglog(1:length(mse_ptyc), mse_ptyc,'-d', ...
             'LineWidth',1.5,'MarkerSize',4,'Color',[1 0.5 0]); % Orange
hold off;

axis square             % square axes again
ax2 = gca;
grid(ax2,'off');
ax2.XMinorGrid  = 'on';
ax2.YMinorGrid  = 'on';
ax2.MinorGridLineStyle = ':';
ax2.GridAlpha        = 0.6;
ax2.MinorGridAlpha   = 0.3;

ax2.XLim = [0.75, num_iterations_error * 1.2];
allE = [mse_joint(:); mse_ptyc(:)];
ax2.YLim = [min(allE) * 0.75, max(allE) * 1.2];

xlabel('Iteration','FontSize',12);
ylabel('MSE','FontSize',12);
title('Error comparison','FontSize',14);
legend([hJ2,hP2], ...
       {'Joint Method','Ptychography Method'}, ...
       'Location','best','FontSize',12);


%%

figure('Units','inches','Position',[1 1 5 5], ...
       'PaperUnits','inches','PaperPosition',[0 0 5 5]);   

hJ = loglog(1:length(f_tot_joint), f_tot_joint, '-o', ...
            'LineWidth',1.5,'MarkerSize',4,'Color',[0 0 1]); % Blue
hold on;
hP = loglog(1:length(f_tot_ptyc), f_tot_ptyc,'-d', ...
            'LineWidth',1.5,'MarkerSize',4,'Color',[1 0.5 0]); % Orange
hold off;

axis square             % force axes to be equal
ax = gca;
grid(ax,'off');         
ax.XMinorGrid  = 'on';  
ax.YMinorGrid  = 'on';
ax.MinorGridLineStyle = ':';
ax.GridAlpha        = 0.6;
ax.MinorGridAlpha   = 0.3;

% pad X so first tick is at 0.75 and last at 1.2×
ax.XLim = [0.75, num_iterations_loss * 1.2];

allL = [f_tot_joint(:); f_tot_ptyc(:)];
ax.YLim = [min(allL) * 0.75, max(allL) * 1.2];

xlabel('Iteration','FontSize',12);
ylabel('Loss Value','FontSize',12);
title('Loss comparison','FontSize',14);
legend([hJ,hP], ...
       {'Joint Method','Ptychography Method'}, ...
       'Location','best','FontSize',12);
