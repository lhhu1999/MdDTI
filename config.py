class Config:
    def __init__(self):
        self.mode = 'gpu'
        self.K_FOLD = 5

        # common
        self.batch_size = 16
        self.atom_dim = 64
        self.residues_dim = 64
        self.weight_decay = 1e-5
        self.lr_decay = 0.5
        self.max_length_components = 30
        self.max_length_skeleton = 150
        self.max_length_residue = 1000
