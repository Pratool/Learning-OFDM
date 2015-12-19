f1 = fopen('../received.dat', 'r');

x = fread(f1, 'float32');

hold all;

st_arr = 9.904e5;
en_arr = 9.919e5;

x_real = x(1:2:end);
x_imag = x(2:2:end);

% plot(x_real);

x_real = x_real(st_arr:en_arr);
x_imag = x_imag(st_arr:en_arr);
[xi, xq, f] = bpsk_timing_sync(x_real, x_imag);
disp(f);

plot(x_real);
plot(xi);
plot(xq);

fclose(f1);