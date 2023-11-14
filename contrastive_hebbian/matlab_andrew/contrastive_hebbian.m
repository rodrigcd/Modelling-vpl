
lambda = .00001;
gamma = 1;


Ny = 1;
Nh = 100;
Ni = 10;

P=50;

W2 = randn(Ny,Nh);
W1 = randn(Nh,Ni);

y = randn(Ny,P);
x = randn(Ni,P);

max_itr = 200;

for itr = 1:max_itr
    
    hf = W1*x;
    yf = W2*hf;

    hc = hf + gamma*W2'*y;
    %xc = x + gamma*W1'*hc;
    
    W1 = W1 + 1/gamma*lambda*(hc*x' - hf*x');
    W2 = W2 + lambda*(y*hc' - yf*hf');
    
    W1 = W1 + 1/gamma*lambda*(W1*x*x' + gamma*W2'*y*x' - W1*x*x');
    W2 = W2 + lambda*(y*(W1*x' + gamma*W2'*y')' - W2*W1*x*(W1*x)');
    
    W1 = W1 + 1/gamma*lambda*(W1*x*x' + gamma*W2'*y*x' - W1*x*x');
    W2 = W2 + lambda*(y*x'*W1' + gamma*y*y'*W2 - W2*W1*x*x'*W1');
    
    W1 = W1 + 1/gamma*lambda*( gamma*W2'*y*x'); % Missing term: -W2'*W2*W1*x*x'
    W2 = W2 + lambda*( (sio - W2*W1*si)*W1' + gamma*y*y'*W2 );
    
    err(itr) = 1/P*norm(y-yf,'fro');

end

plot(err)


