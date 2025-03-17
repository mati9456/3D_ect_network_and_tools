function remove_duplicates = remove_duplicates(c)
    consts = PreprocessingConsts;
    c = c(consts.repeating);
    c= reshape(c, [1, 496]);
    remove_duplicates = c;
end