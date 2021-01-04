close all;
clear all;
clc;

pkg load signal;

fs = 22050;  # arbitrary sample rate
f0 = 100;    # pulse train sample rate
w = 0.3/f0;  # pulse width 1/10th the distance between pulses
x = pulstran (0:1/fs:4/f0, 0:1/f0:4/f0, "rectpuls", w);

Fs=100; #Samplingrate

figure(2);
plot ([0:length(x)-1]*1000/fs, x, "bo-");

step=ceil(20*Fs/1000);    # one spectral slice every 20 ms
window=ceil(200*Fs/1000); # 100 ms data window

## test of automatic plot
figure(1);
[S, f, t] = specgram(x);
specgram(x, 2^nextpow2(window), Fs, window, window-step);

yt = get(gca, 'YTick');
set(gca, 'YTick',yt, 'YTickLabel',yt*1E+4);

#figure(3);
#x = [-1 0 1 2 3 4];
#y = [0 0 1 1 0 0 ];
#plot(x,y,"bo-");

