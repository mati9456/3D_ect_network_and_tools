function reconstruction = reconstructTorch(net, input)
    consts = PreprocessingConsts;
    input = input(consts.repeating);
    input= reshape(input, [1, 496]);
    input = (input - consts.empty)./ (consts.full - consts.empty);
    %input = (input - 0.10184967) ./ 0.14424020; % tutaj normowanie wejscia
    reconstruction = net.predict(input);
    reconstruction = (reconstruction.*1)+2;
end