function [noise,amplitude,freq, phase] = GenerateNoise(t)

noise = 0;
range = 24;
freq = linspace(0,50,range);%normrnd(1, 50 ,[1,range]);
phase = linspace(0,pi,range);%normrnd(0,pi,[1,range]);
amplitude = linspace(0,0.01,range);%normrnd(0,1,[1, range])/100;
    for i = 1:24
        
        noise = noise + amplitude(i)*sin(freq(i)*t+phase(i));
    end
    
end