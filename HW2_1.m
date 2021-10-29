%% (a) Gaussian process Regression of Sample data

data = [0, -.00799366; .6981317, .50389564; 1.3962634, .92653312; 2.0943951, ...
    .7628036; 2.7925268, .37189376; 3.4906585, -.1965461; 4.1887902, -.93017225; ...
    4.88692191, -1.04932639; 5.58505361, -.72417058; 6.28318531, -.07469816];


%assume y = f(x) + epsilon, where f is gaussian with mean m and variance k.
x = data(:,1);
y = data(:,2);
%Plot data
scatter(x,y);
hold on;
%Normalize data by subtracting mean
[nrows,ncols] = size(data);
%n = nrows*ncols;
%mean = sum(sum(data))/n;
%data = data - mean;

%epsilon is N(0,.01):
stdev = .01;

%make cov matrix: 10 data points so cov matrix should be 10 x 10
K = zeros(nrows);

%Make K
for i=1:nrows
    for j=i:nrows
        x_1 = data(i,1); x_2 = data(j,1);
        K(i,j) = exp(-(x_1-x_2)^2);
        K(j,i) = K(i,j);
    end
end

%% prediction with m(x) = 0

%initialize prediction vector 1
y_hat = zeros(nrows,1);
%calc (K(x,x) + stdev*I(10))^(-1)
K_inv = inv(K + stdev*eye(nrows)); 
alpha = K_inv*y;

for i=1:nrows
   y_hat(i,:) = alpha.'*K(:,i);
end

plot(x, y_hat);
hold on;

%% prediction with m(x) = sin(x)

y_hats = sin(x) + K*K_inv*(y - sin(x));

plot(x, y_hats);
legend('y', 'GPR with m(x)=0', 'GPR with m(x) = sin(x)');
title('Performance of GPR with different mean functions')    
ylabel('y predictions');
xlabel('x');
hold off;

%% (b) make cross covariance matrix using K_xstarx

%need to compute posterior at 100 locations
x_star = linspace(0,2*pi,100);
len = size(x_star,2);

%make K_xstarx
K_xstarx = zeros(len,nrows);

for i=1:len
    for j=1:nrows
        x_1 = x_star(i); x_2 = x(j);
        K_xstarx(i,j) = exp(-(x_1-x_2)^2);
    end
end

%make K_xstarxstar
K_xstarxstar = zeros(len,len);

for i=1:len
    for j=i:len
        x_1 = x_star(i); x_2 = x_star(j);
        K_xstarxstar(i,j) = exp(-(x_1-x_2)^2);
        K_xstarxstar(j,i) = K_xstarxstar(i,j);
    end
end

new_Kinv = inv(K);
%new cov matrix = K(x_*,x_*) - K(x_*,x)K(x,x)^(-1)K(x,x_*). This is 100x100 new cov matrix
K_cond = K_xstarxstar - K_xstarx*new_Kinv*(K_xstarx.');

K_cond = K_cond + 10^(-10)*eye(100)

%K_cond = WW^T so we factorize 
W = chol(K_cond, 'lower'); 

%% now we get 20 draws and plot them. 

for i=1:20 
    z = randn(len,1) %random variable N(0,1)
    y_new = W*z; %draws from posterior distribution 
    plot(x_star, y_new)
    hold on;
end

title('Evaluating 20 draws from distribution')    
ylabel('predicted values');
xlabel('x');
hold off;


        