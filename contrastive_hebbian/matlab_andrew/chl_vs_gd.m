clear

plot(0:.1:1,0:.1:1,'r--'),hold on
gammas = [-20:20]%:.1:1];%-.69;
for gi = 1:length(gammas)
    delta_input = 5;
    sigma_L1 = 16;
    gamma = gammas(gi);
    targ = 1;

    tau = 1;
    dt = .005;
    Nitr = 5000;

    N1 = 101;
    ang = linspace(-90,90,N1);

    gp = normpdf(ang,delta_input/2,sigma_L1)';
    gm = normpdf(ang,-delta_input/2,sigma_L1)';
    gp = gp/max(gp);%;/norm(gp);
    gm = gm/max(gm);%;/norm(gm);

    d = gp - gm;

    v = d'*gp;

    alpha = 0;
    beta = 0;

    s = zeros(Nitr,2);
    for t = 1:Nitr


        bdot = 1/tau*((targ - v*beta*(1 + 2*alpha))*(1 + 2*alpha) + 2*gamma*beta*(targ.^2 - (v*beta*(1 + 2*alpha)).^2));
        adot = 1/tau*beta*(targ - v*beta*(1 + 2*alpha));


        alpha = alpha + adot*dt;
        beta = beta + bdot*dt;

        s(t,:) = [alpha beta];

    end
    plot(s(:,1),s(:,2),'linewidth',2)
    axis equal
    hold on
    
    layer_ratio(gi) = beta/alpha;
    
end

%%

plot(gammas,layer_ratio)


%%

%(2*t + 2*gamma*targ*x + 2*gamma*v*x^2 + 4*gamma*v*x^2*t + 1)/x
%(2*t + 2*gamma*targ*x + (2*gamma*v + 4*gamma*v*t)*x^2 + 1)/x

%(1 + 2*t)*x^-1 + 2*gamma*targ + (2*gamma*v + 4*gamma*v*t)*x

%take it in pieces? suppose gamma = 0. Then have
%db*x = (1+2*t)dt or 1/2x^2 = t + t^2 so f(t) = sqrt(2*(t + t^2)). Then 

%dx/dt = 1/2*(2*(t + t^2))^(-1/2)*2*(1 + 2t) = (2*(t + t^2))^(-1/2)*(1 + 2t)=(1+2*t)/f(t), works!

%maybe f(t) = sqrt(2*(t + t^2)) + 2*gamma*targ*t. I think that works.

%%
syms t gamma targ v

f = sqrt(2*(t + t^2)) + 2*gamma*targ*t + C*exp(2*gamma*v*(t + t^2));
diff(f,t)


(2*gamma*v + 4*gamma*v*t)*x
log(x) = 2*gamma*v*(t + t^2) + C

g = C*exp(2*gamma*v*(t + t^2))

