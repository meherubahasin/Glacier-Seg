class GlacierHDF5PatchDataset3(Dataset):
    def __init__(self, hdf5_file_path, patch_size=PATCH_SIZE, target_size=TARGET_SIZE, length = 600):
        self.hdf5 = h5py.File(hdf5_file_path, 'r')
        self.tiles = [name for name in self.hdf5.keys() if all(m in self.hdf5[name] for m in MODALITIES_3)]
        self.patch_size = patch_size
        self.target_size = target_size
        self.length = length

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        tile_name = random.choice(self.tiles)
        tile = self.hdf5[tile_name]
        h, w = tile[MODALITIES_3[0]].shape[:2]
        y = random.randint(0, h - self.patch_size)
        x = random.randint(0, w - self.patch_size)

        input_channels = []
        for key in MODALITIES_3:
            arr = tile[key][y:y+self.patch_size, x:x+self.patch_size, :]
            arr = normalize(arr)
            input_channels.append(arr)
        input_patch = np.concatenate(input_channels, axis=2)
        label = tile[LABEL_KEY][y:y+self.patch_size, x:x+self.patch_size]
        if label.ndim == 3:
            label = label[:, :, 0] if label.shape[2] == 1 else np.argmax(label, axis=2)
        input_patch, label = augment_patch(input_patch, label)
        input_tensor = torch.tensor(np.ascontiguousarray(input_patch)).permute(2, 0, 1).float()
        label_tensor = torch.tensor(np.ascontiguousarray(label), dtype=torch.long)
        return input_tensor, label_tensor

    def close(self):
        self.hdf5.close()
