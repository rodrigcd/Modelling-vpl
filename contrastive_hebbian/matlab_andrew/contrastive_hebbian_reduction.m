clear all

delta_input = 5;
sigma_L1 = 16;
targ = 1;

tau = 1;
dt = .005;
Nitr = 800;

N1 = 101;
ang = linspace(-90,90,N1);

gp = normpdf(ang,delta_input/2,sigma_L1)';
gm = normpdf(ang,-delta_input/2,sigma_L1)';
gp = gp/max(gp);%;/norm(gp);
gm = gm/max(gm);%;/norm(gm);

d = gp - gm;

u = d'*d;
v = d'*gp;

sig = sigma_L1;
da = median(diff(ang));

dprime = delta_input/sig

v2 = 1/(2*sqrt(pi)*sig)*(1-exp(-(dprime/2)^2))
v*da


W{1} = [gp gm];
               
W{2} = zeros(1,N1);

alpha = 0;
beta = 0;

%%
s = zeros(Nitr,2);
for t = 1:Nitr


    Wdot{1} = 1/tau*W{2}'*([targ -targ] - W{2}*W{1});
    Wdot{2} = 1/tau*([targ -targ] - W{2}*W{1})*W{1}';

    W{1} = W{1} + Wdot{1}*dt;
    W{2} = W{2} + Wdot{2}*dt;

    bdot = 1/tau*(targ - v*beta - u*alpha*beta)*(1 + 2*alpha);
    adot = 1/tau*beta*(targ - v*beta - u*alpha*beta);

    % Idea: rewrite eqn as a vs b, no dt.

    % da/dt / db/dt = da/db = (1 + 2*beta)/alpha
    % alpha da = (1 + 2*beta)db
    % alpha^2/2 = beta + beta^2 + C
    %
    % => so a^2/2 - b - b^2 is an invariant of motion
    %
    % hence if you start from (0,0), C = 0. Then alpha = (2b + 2b^2)^1/2


    alpha = alpha + adot*dt;
    beta = beta + bdot*dt;

    s(t,:) = [alpha beta];
    semp(t,:) = [W{2}*d/(d'*d) (W{1}(:,1)-gp)'*d/(d'*d)];
end

plot(ang,gm,ang,gp,'linewidth',2)
xlabel('Angle (deg)')
ylabel('Response')

