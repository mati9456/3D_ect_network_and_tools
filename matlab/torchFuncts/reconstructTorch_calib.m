function reconstruction = reconstructTorch_calib(net, input, minC, maxC)
    consts = PreprocessingConsts;
    input = input(consts.repeating);
    input= reshape(input, [1, 496]);
    minC2 = minC(consts.repeating);
    maxC2 = maxC(consts.repeating);
    input = (input - minC2)./ (maxC2 - minC2);
    %input = (input - 0.10184967) ./ 0.14424020; % tutaj normowanie wejscia
    reconstruction = net.predict(input);
    reconstruction = (reconstruction.*1)+2;
end