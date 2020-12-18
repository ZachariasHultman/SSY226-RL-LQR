function [noise,amplitude,freq, phase] = GenerateNoise(t)

noise = 0;
range = 24;
freq = linspace(0,50,range);
phase = linspace(0,pi,range);
amplitude = linspace(0,0.05,range);

    for i = 1:range
        noise = noise + amplitude(i)*sin(freq(i)*t+phase(i));
    end
    
end