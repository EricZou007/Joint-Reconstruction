function [tv, grad_u] = tv_complex(u, tv_type, eps_tv)

[nr, nc] = size(u);

dx = [u(:,2:end) - u(:,1:end-1), zeros(nr,1)];
dy = [u(2:end,:) - u(1:end-1,:); zeros(1,nc)];

switch lower(tv_type)
    case 'iso'
        % isotropic: sum sqrt(|dx|^2 + |dy|^2 + eps^2)
        denom = sqrt(abs(dx).^2 + abs(dy).^2 + eps_tv^2);
        tv = sum(denom(:));

        px = dx ./ denom;
        py = dy ./ denom;

    case 'aniso'
        % anisotropic: sum ( sqrt(|dx|^2 + eps^2) + sqrt(|dy|^2 + eps^2) )
        denomx = sqrt(abs(dx).^2 + eps_tv^2);
        denomy = sqrt(abs(dy).^2 + eps_tv^2);
        tv = sum(denomx(:)) + sum(denomy(:));

        px = dx ./ denomx;
        py = dy ./ denomy;

    otherwise
        error('tv_type must be ''iso'' or ''aniso''.');
end

% Divergence (adjoint of forward diff with zero boundary)
divx = [px(:,1), px(:,2:end) - px(:,1:end-1)];
divy = [py(1,:); py(2:end,:) - py(1:end-1,:)];

% Gradient of TV: -div(px, py)
grad_u = -(divx + divy);
end