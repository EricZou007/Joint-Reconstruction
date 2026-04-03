function [xnew, fnew, gnew, nf1, ierror, alpha1, stepNorm, varargout] = ...
    lin1 (p, x, f, alpha, g, sfun)
%---------------------------------------------------------
% line search (naive)
%---------------------------------------------------------
% set up
%---------------------------------------------------------
ierror = 2; % changed here, used to be 3
xnew   = x;
fnew   = f;
gnew   = g;
maxit  = 10;
stepNorm = 0;
%%%%%%%%%%%%%%%%%%%#####fix step length as 1 ##############################
%   alpha=1;
%%%%%%%%%%%%%%%%%%%#######################################################
% disp('### In LINE SEARCH: lin1 ###')
% fprintf(' g''p = %e\n',g'*p);
if (alpha == 0); ierror = 0; maxit = 1; end;
alpha1 = alpha;
q0=p'*g;
%---------------------------------------------------------
% line search
%---------------------------------------------------------
% al=linspace(-1,1,3000);
% for i=1:3000,ftemp(i)=feval(sfun,x+al(i)*p);end
% figure, plot(al,ftemp,'r.-');
c1 = 1e-4; 
c2 = 0.9;

for itcnt = 1:maxit;
    xt = x + alpha1.*p;




    %%%%%%%%%%%#############################################################
    [ft, gt] = feval (sfun, xt);
%   Armijo =ft<f+c1*alpha1*q0;
%   Wolfe = abs(p'*gt)<c2*abs(q0);
    Wolfe = p'*gt >= c2*q0;
    if (ft < f)% Wolfe)%
%         fprintf('using Armijo...\n');
        ierror = 0;
        xnew   = xt;
        fnew   = ft;
        gnew   = gt;
        stepNorm   = norm(xnew(:)-x(:));
 
        break;
    end
    alpha1 = alpha1 ./ 2;
end
if (ierror == 3)
    % al=linspace(-4,4,100);
    % for i=1:100,ftemp(i)=feval(sfun,x+al(i)*p);end
    % figure, plot(al,ftemp,'r.-');
    alpha1 = 0;
end
nf1 = itcnt;

% if (nargout == 7)
if nargout==8
    dfdp = dot(gt, p);
    varargout{1} = dfdp;
end
