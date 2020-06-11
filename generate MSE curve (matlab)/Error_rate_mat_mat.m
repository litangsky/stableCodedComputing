x = [80, 90, 100, 110, 120];
y1 = [2.439798241177374e-08,2.444171389801081e-09,2.447346076440714e-10,2.433704568581326e-11,2.447249190915922e-12];
y2 = [2.459288717275408e-08,2.449032452496838e-09,2.450839775679502e-10,2.439191896021262e-11,2.439609007888255e-12];
y3 = [0.000970865615627419,9.66987830950721e-05,9.66407287973113e-06,9.68663926463043e-07,9.71756172292562e-08];
y4 = [0.000115712992513099,1.14467466739739e-05,1.15077419763111e-06,1.15614808184377e-07,1.14482279983689e-08];
semilogy(x,y1, '-ro', x, y2, '-kx', x, y3, '-m+', x, y4, '-gs', 'MarkerSize',15, 'linewidth',2);
lgd = legend( 'Complex Vand.', 'Rot. Mat. Embed.', '(Fahim & Cadambe, 2019)', '(Subramaniam et al., 2019)');
lgd.FontSize = 12;
grid on;
xlabel('SNR (dB)','fontsize',15);
ylabel('Normalized MSE (worst case)', 'fontsize', 14);