close all;
clear all;
clc;

pkg load signal;

Start=0;
Schrittweite=0.00001;
Ende=1.2;

t=Start:Schrittweite:Ende;

Fs=2000;

#chirp (t0, f0, t1, f1)
#chirp(t, f0 = 0, t1 = 1, f1 = 100, form = c("linear"), phase = 0)
#x=chirp(t0, f0, t1, f1, form = "linear", phase = 0);

#x = chirp(t,Start,Schrittweite,Ende);  # freq. sweep from 0-500 over 2 sec.
#save -ascii C:\_Design\emv\data.txt

x = 0;
x = load('-ascii','C:\_Design\emv\dataLT.txt');

step=ceil(20*Fs/1000);    # one spectral slice every 20 ms
window=ceil(100*Fs/50000); # 100 ms data window

## test of automatic plot
figure(1);
[S, f, t] = specgram(x);
specgram(x, 2^nextpow2(window), Fs, window, window-step);
figure(2);
plot(x);


