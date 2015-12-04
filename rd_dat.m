f1 = fopen('bar.dat', 'r');

x = fread(f1, 'float32');

x_real = x(1:2:end);
x_imag = x(2:2:end);

plot(x_real);

fclose(f1);