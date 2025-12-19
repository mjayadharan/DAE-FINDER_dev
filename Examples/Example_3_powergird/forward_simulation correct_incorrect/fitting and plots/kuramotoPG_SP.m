function dtheta = kuramotoPG_SP(t,theta,N,ngi,ngt,nl,wref,K,H,A,D,gamma,xeq,input)
% Structure-preserving model of a power grid (network of coupled oscillators)
%
% Robust to row/column inputs:
%   - Converts everything to column vectors internally
%   - Ensures xeq has the same length as theta
%   - Returns a COLUMN vector (n x 1)

    % --- Normalize shapes (columns) ---
    theta = theta(:);          % n x 1
    xeq   = xeq(:);            % maybe n x 1, maybe not
    input = input(:);          % n x 1

    % If xeq length doesn't match theta, ignore it (use zeros)
    if numel(xeq) ~= numel(theta)
        xeq = zeros(size(theta));
    end

    theta = theta + xeq;       % now safe

    % Sizes
    % n  = 3*ngi + nl;         % not needed explicitly, but implied
    % N  = 2*ngi + nl;         % given

    % States
    x1 = theta(1:ngi);                        % generator phase
    x2 = theta(ngi+1:2*ngi);                  % generator frequency
    x3 = theta(2*ngi+1:2*ngi+ngt+nl);         % load phase

    % Coupling term
    ph  = [x1; x3];                           % N x 1
    aux = sum( sin( ones(N,1)*ph.' - ph*ones(1,N) + gamma).*K , 2 );

    % Generator dynamics: 2nd order Kuramoto
    dx1 = x2;
    dx2 = wref./(2*H(1:ngi)).*( A(1:ngi) - (D(1:ngi)/wref).*x2 + aux(1:ngi,1) );

    % Load dynamics: 1st order Kuramoto
    dx3 = wref./D(ngi+1:end).*( A(ngi+1:end) + aux(ngi+1:end,1) );

    % Return n x 1
    dtheta = [dx1; dx2; dx3] + input; 
end
