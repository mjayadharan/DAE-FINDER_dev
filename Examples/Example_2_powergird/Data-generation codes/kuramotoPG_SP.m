function dtheta = kuramotoPG_SP(t,theta,N,ngi,ngt,nl,wref,K,H,A,D,gamma,xeq,input)
% Structure-preserving model of a power grid (network of coupled
% oscillators)
theta = theta + xeq';
x1(:,1) = theta(1:ngi);                     % generator phase
x2(:,1) = theta(ngi+1:2*ngi);               % generator frequency
x3(:,1) = theta(2*ngi+1:2*ngi+ngt+nl);      % load phase
aux = sum( sin( ones(N,1)*[x1; x3]' - [x1; x3]*ones(1,N) + gamma).*K ,2) ;

% Generator dynamics: 2nd order Kuramoto
dx1 = x2;
dx2 = wref./(2*H(1:ngi)).*( A(1:ngi) - (D(1:ngi)/wref).*x2 + aux(1:ngi,1) );

% Load dynamics: 1st order Kuramoto
dx3 = wref./D(ngi+1:end).*( A(ngi+1:end) + aux(ngi+1:end,1) );

% Return
dtheta = [dx1; dx2; dx3] + input; 
end



