% Multi-hypothesis KF code

% State:
% Our state is a 12x1 vector

% Dynamics Models:
% We have 7 dynamics models
% 1 - G1: Reaching for suture with right hand
% 2 - G11: Dropping suture and moving to next knot
% 3 - G12: Reaching for suture with left hand
% 4 - G13a: Making C loop front
% 5 - G13b: Making C loop back
% 6 - G14: Reaching for suture with right hand
% 7 - G15: Pulling suture with both hands
%
% clear
clc
close all
clear state M_new label M_interp
%load('data.mat')
%load('Q.mat')

A = {G1_A,G11_A,G12_A,G13_A,G14_A,G15_A};
Q = {G1_Q;G11_Q;G12_Q;G13_Q;G14_Q;G15_Q};

num_features = 12;

%our measurement matrix is just the identity
C = eye(num_features);

[label,M_new] = read_data('C',2);
tfinal = length(M_new);
t = 0:1/30:(tfinal-1)/30;

if resample == 1
    t_int = 0:1/100:(tfinal-1)/30;
    for j = 1:size(M_new,2)
        M_interp(:,j) = interp1(t,M_new(:,j),t_int);
    end
    M_new = M_interp;
end

% Initialize our states as cells
dt = 1/30;


for i = 1:6
    
    miu{i,1} = M_new(1,:)';
    sigma{i,1} = 1.*eye(num_features);
    phi(i,1) = 1/6;
    Q{i,1} = 1*eye(num_features); %the rest of the gestures  such a
%      Q{2,1} = 0.2*eye(num_features); %the rest of the gestures  such a
%      Q{3,1} = 0.0001*eye(num_features); %the rest of the gestures  such a
%      Q{4,1} = 0.0001*eye(num_features); %the rest of the gestures  such a
%      Q{5,1} = 0.001*eye(num_features); %the rest of the gestures  such a
    
    R{i,1} = 1.*eye(num_features,num_features);
    
end
state(1) = 3;

for i = 2:length(M_new)
    
    for j = 1:6
        %do estimation
        [miu{j,i},sigma{j,i},phi(j,i)] = mhkf(M_new(i,:)',miu{j,i-1},sigma{j,i-1},phi(j,i-1),A{j},C,Q{j},R{j});
    end
    
    %renormalize phis
    phi(:,i) = phi(:,i)./sum(phi(:,i));
    
    %record our predicted state
    state(i) = find(phi(:,i)== max(phi(:,i)));
    if state(i) >= 4
        state(i) = state(i)+1;
    end
    
    
end

t_label = 0:1/30:(length(label)-1)/30;
plot(t_label,label,'LineWidth',2)
hold on
if resample ==1
    plot(t_int,state);
else
    plot(t,state);
end

function [miu,sig,phi] = mhkf(z,m,s,p,A,C,Q,R)
%MHKF main function. Takes in the observation z, previous miu m, sigma s,
%previous posterior p, the A,C,Q and R matrices.
%The function outputs the updated miu, sigma and the unnormalized posterior
%phi.

m_pred = A*m;
s_pred = A*s*A'+Q;

H = R + C*s_pred*C';
K = s_pred*C'*inv(H);

n = length(m_pred);

miu = m_pred + K*(z-eye(n)*m_pred);
sig = (eye(n)-K*C)*s_pred;

Sy = R+C*sig*C';

try
    mvnpdf(z,C*miu,Sy);
catch ME % 
     %if not positive semidefinite, find the nearest using nearestSPD function 
     %https://www.mathworks.com/matlabcentral/fileexchange/42885-nearestspd
     Sy = nearestSPD(Sy);
     H = nearestSPD(H);
end

phi = mvnpdf(z,miu,Sy)*p;
%phi = mvnpdf(z,miu,H)*p;
end