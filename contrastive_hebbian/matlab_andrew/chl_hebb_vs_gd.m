clear
tic
%plot(0:.1:1,0:.1:1,'r--'),hold on

delta_input = 5;
sigma_L1 = 16;
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

plot(ang,gp,ang,gm,ang,gp-1*d,ang,gm+.25*d,'linewidth',2)

%%

gammas = [-2:.01:2];%-.69;
etas = -1:.01:1;
for etai = 1:length(etas)
    eta = etas(etai);
    for gi = 1:length(gammas)

        gamma = gammas(gi);

        alpha = 0;
        beta = 0;
        chi = 0;

        s = zeros(Nitr,4);
        for t = 1:Nitr


            bdot = 1/tau*((targ - v*beta*(1 + chi + 2*alpha))*(1 + chi + 2*alpha) + 2*gamma*beta*(targ.^2 - (v*beta*(1 + chi + 2*alpha)).^2));
            adot = 1/tau*beta*(targ - v*beta*(1 + chi + 2*alpha)) + 1/tau*eta*(targ - v*beta*(1 + chi + 2*alpha)).^2*alpha;
            chidot = 1/tau*eta*(targ - v*beta*(1 + chi + 2*alpha)).^2*(1+chi);

            e = (targ - v*beta*(1 + chi + 2*alpha));

            alpha = alpha + adot*dt;
            beta = beta + bdot*dt;
            chi = chi + chidot*dt;

            s(t,:) = [alpha beta chi e];

        end
    %     plot(s(:,1),s(:,2),'linewidth',2)
    %     axis equal
    %     hold on
        alphge(gi,etai) = alpha;
        betage(gi,etai) = beta;
        chige(gi,etai) = chi;
        ege(gi,etai) = e;
        
    end
end
plot(s)

%%
[gx,ex] = meshgrid(gammas,etas);

surf(gx,ex,chige'+2*alphge')

%%
surf(gx,ex,ege')

%%

slope_ratio = betage./(chige+2*alphge);
surf(gx,ex,slope_ratio')
xlabel('\gamma')
ylabel('\eta')

%%

for gi = 1:length(gammas)
    ind = find(slope_ratio(gi,:)<1,1);
    if ~isempty(ind) 
        eta_slope_change(gi) = etas(ind);
    else
        eta_slope_change(gi) = nan;
    end
end
plot([-2; 2],[0 ;0],'k',[0; 0],[1 -1],'k'),hold on
plot(gammas,eta_slope_change,'linewidth',4)
set(gca,'FontSize',18)
ylabel('Hebbian component \eta')
xlabel('Top-down feedback \gamma')

xlim([-1.8 1.8])
ylim([-.8 .8])

%
for gi = 1:length(gammas)
    ind = find(slope_ratio(gi,:)<3,1);
    if ~isempty(ind) 
        eta_slope_change(gi) = etas(ind);
    else
        eta_slope_change(gi) = nan;
    end
end
hold on
plot(gammas,eta_slope_change,'linewidth',4)


%
pos_cutoff = .1;
neg_cutoff = -.3;

for gi = 1:length(gammas)
    ind = find(chige(gi,:)>pos_cutoff,1);
    ind2 = find(chige(gi,:)>neg_cutoff,1);
    if ~isempty(ind) 
        eta_pos(gi) = etas(ind);
    else
        eta_pos(gi) = nan;
    end
    if ~isempty(ind2) 
        eta_neg(gi) = etas(ind2);
    else
        eta_neg(gi) = nan;
    end
    
end
hold on
plot(gammas,eta_neg,gammas,eta_pos,'linewidth',3)
toc
