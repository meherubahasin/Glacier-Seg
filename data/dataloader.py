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



class GlacierSAR(Dataset):
    def __init__(self, root, tile_size=512, augment=False, duplicate_channels=True):
        self.root = Path(root)
        self.duplicate = duplicate_channels
        self.tile_size = tile_size

        self.im_dir = self.root / "sar"
        self.mask_dir = self.root / "zone"
        self.files = sorted(f.stem for f in self.im_dir.glob("*.png"))

        # --- Transformations ---
        if augment:
            self.cropper = A.Compose([
                A.PadIfNeeded(tile_size, tile_size, border_mode=cv2.BORDER_REFLECT),
                A.RandomCrop(tile_size, tile_size),
            ])
        else:
            self.cropper = A.Compose([
                A.PadIfNeeded(tile_size, tile_size, border_mode=cv2.BORDER_REFLECT),
                A.CenterCrop(tile_size, tile_size),
            ])

        self.to_tensor = A.Compose([
            A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
            ToTensorV2()
        ])

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        stem = self.files[idx]

        # Load grayscale image
        img = cv2.imread(str(self.im_dir / f"{stem}.png"), cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise FileNotFoundError(f"Image missing: {self.im_dir / f'{stem}.png'}")

        # Load mask
        mask_path = self.mask_dir / f"{stem}.png"
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise FileNotFoundError(f"Mask missing: {mask_path}")

        # Binarize mask (0 or 1)
        mask = (mask > 127).astype("uint8")

        # Apply deterministic crop/pad
        out = self.cropper(image=img, mask=mask)
        img, mask = out["image"], out["mask"]

        # Duplicate grayscale -> 3 channels if requested
        if self.duplicate:
            img = np.stack([img] * 3, axis=-1)  # [H,W,3]
        else:
            img = np.expand_dims(img, axis=-1)  # [H,W,1]

        # Convert to tensor
        out = self.to_tensor(image=img, mask=mask)
        img_tensor = out["image"].float()     # [3,H,W] or [1,H,W]
        mask_tensor = out["mask"].long()      # [H,W]

        # Expand mask to [1,H,W]
        mask_tensor = mask_tensor.unsqueeze(0)

        return img_tensor, mask_tensor



# Channel name mapping (15 channels in HKH dataset)
CHANNELS = {
    0: "blue",
    1: "green",
    2: "red",
    3: "nir",
    4: "swir1",
    5: "thermal_low",
    6: "thermal_high",
    7: "swir2",
    8: "pan",
    9: "quality",
    10: "ndvi",
    11: "ndsi",
    12: "ndwi",
    13: "elevation",
    14: "slope"
}


class GlacierDatasetHKH(Dataset):
    def __init__(self, img_paths, mask_paths, channels=None, augment=True, length=None):
        """
        Args:
            img_paths (list): List of image tile .npy paths
            mask_paths (list): List of mask tile .npy paths (aligned with img_paths)
            channels (list): Which channels to use (can be indices [0,1,2] or names ["red","nir"])
            augment (bool): Whether to apply augmentations
            length (int): Limit number of samples to load (for quick testing/debugging)
        """
        assert len(img_paths) == len(mask_paths), "Image and mask list must match in length."

        # If user specifies length, truncate dataset
        if length is not None and length < len(img_paths):
            img_paths = img_paths[:length]
            mask_paths = mask_paths[:length]

        self.img_paths = img_paths
        self.mask_paths = mask_paths
        self.augment = augment

        # Handle channel selection (allow names or indices)
        if channels is None:
            self.channels = list(range(15))  # default: all channels
        else:
            self.channels = [
                i if isinstance(i, int) else {v: k for k, v in CHANNELS.items()}[i]
                for i in channels
            ]

        # --- Augmentation pipeline ---
        self.transform = A.Compose([
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.Affine(scale=(0.9, 1.1), shear=(-10, 10), rotate=(-20, 20), p=0.3),
            A.RandomBrightnessContrast(p=0.3),
            A.GaussNoise(std_range=(0.01, 0.05), mean_range=(0, 0), per_channel=True, p=0.3),
            A.GaussianBlur(blur_limit=3, p=0.2),
        ])

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        # Load image + mask
        img = np.load(self.img_paths[idx])
        mask = np.load(self.mask_paths[idx])

        # If mask is one-hot, reduce to single channel
        if mask.ndim == 3 and mask.shape[-1] > 1:
            mask = np.argmax(mask, axis=-1)

        # Channel selection
        img = img[:, :, self.channels]

        # Normalization (per channel min-max 0–1)
        img = (img - img.min(axis=(0, 1), keepdims=True)) / (
            img.max(axis=(0, 1), keepdims=True) - img.min(axis=(0, 1), keepdims=True) + 1e-6
        )

        # Apply augmentations
        if self.augment:
            transformed = self.transform(image=img, mask=mask)
            img, mask = transformed["image"], transformed["mask"]

        # Convert to torch tensors
        img = torch.from_numpy(img.transpose(2, 0, 1)).float()  # (C, H, W)
        mask = torch.from_numpy(mask).long()  # (H, W)

        return img, mask

