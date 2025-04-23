v2 model: Using pre-processing specs as defined in Audio-Visual paper
v3 model: Expanded BiLSTM hidden dim to 1024
v4 model: Switch to LayerNorm between BiLSTM layers (using old hidden dim of 417), and used small learning rate 1e-4
v5 model: Same as v4, but with hidden dim 1024