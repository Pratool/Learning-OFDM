f1 = fopen('send.dat', 'w');

x_real = ones(1e6, 1)*0.7;
x_imag = zeros(1e6,1);

x = zeros(2*length(x_real),1);

x(1:2:end) = x_real;
x(2:2:end) = x_imag;

fwrite(f1, x, 'float32');

fclose(f1);


