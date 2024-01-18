import torch
import os
from PIL import Image
from torchvision.datasets.utils import download_and_extract_archive
import concurrent.futures
import csv
import hashlib
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

# https://github.com/ricevutoriccardo/Federated-Learning-with-ResNet-50-on-CIFAR-10/blob/c93a50bc4c29751f1e054a81b77c819c97ad3444/gradientInversionattack/breaching-main/breaching/cases/data/datasets_vision.py#L353
class Birdsnap(torch.utils.data.Dataset):
    """This is the BirdSnap dataset presented in
    - Berg et al., "Birdsnap: Large-scale Fine-grained Visual Categorization of Birds"
    It contains a lot of classes of birds and can be used as a replacement for ImageNet validation images
    with similar image fidelity but less of the baggage, given that all subjects are in fact birds.
    This is too small to train on though and hence not even partitioned into train/test.
    Several images are missing from flickr (in 2021), these will be discarded automatically.
    """

    METADATA_URL = "http://thomasberg.org/datasets/birdsnap/1.1/birdsnap.tgz"
    METADATA_ARCHIVE = "birdsnap.tgz"
    META_MD5 = "1788158175f6ae794aebf27bcd7a3f5d"
    BASE_FOLDER = "birdsnap"

    def __init__(self, root, split="train", transform=None, target_transform=None, download=True, crop_to_bbx=False):
        """Init with split, transform, target_transform."""
        self.root = os.path.expanduser(root)
        self.transform = transform
        self.target_transform = target_transform
        self.crop_to_bbx = crop_to_bbx  # Crop to dataset default bounding boxes
        self.split = split
        if download:
            self.download()
        if not self.check_integrity():
            raise ValueError("Dataset Birdsnap not downloaded completely or possibly corrupted.")

        self._purge_missing_data()

    def _check_integrity_of_metadata(self, chunk_size=8192):
        """This only checks if all files are there."""
        try:
            with open(os.path.join(self.root, self.METADATA_ARCHIVE), "rb") as f:
                archive_hash = hashlib.md5()
                while chunk := f.read(chunk_size):
                    archive_hash.update(chunk)
            return self.META_MD5 == archive_hash.hexdigest()
        except FileNotFoundError:
            return False

    def check_integrity(self):
        """Full integrity check."""
        if not self._check_integrity_of_metadata():
            return False
        else:
            self._parse_metadata()
            missing_images = 0
            for idx, file in enumerate(self.meta):
                if not self._verify_image(idx):
                    missing_images += 1
            if missing_images > 0:
                print(f"{missing_images} images could not be downloaded.")
            return True

    def download(self):
        # Metadata:
        if self._check_integrity_of_metadata():
            print("Metadata already downloaded and verified")
        else:
            download_and_extract_archive(self.METADATA_URL, self.root, filename=self.METADATA_ARCHIVE)
        # Actual files:
        self._parse_metadata()

        missing_ids = []
        for idx, file in enumerate(self.meta):
            if not self._verify_image(idx):
                missing_ids.append(idx)
        if len(missing_ids) > 0:
            print(f"Downloading {len(missing_ids)} missing files now...")
            self.scrape_images(missing_ids)

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        """Return image, label."""
        img = Image.open(self.paths[index])
        if self.crop_to_bbx:
            img = img.crop(
                (
                    self.meta[index]["bb_x1"],
                    self.meta[index]["bb_y1"],
                    self.meta[index]["bb_x2"],
                    self.meta[index]["bb_y2"],
                )
            )
        img = img.convert("RGB")
        label = self.labels[index]

        img = self.transform(img) if self.transform else img
        label = self.target_transform(label) if self.target_transform else label
        return img, label

    def _parse_metadata(self):
        """Metadata keys are
        dict_keys(['url', 'md5', 'path', 'species_id', 'bb_x1', 'bb_y1', 'bb_x2', 'bb_y2', 'back_x', 'back_y', 'beak_x',
        'beak_y', 'belly_x', 'belly_y', 'breast_x', 'breast_y', 'crown_x', 'crown_y', 'forehead_x', 'forehead_y',
        'left_cheek_x', 'left_cheek_y', 'left_eye_x', 'left_eye_y', 'left_leg_x', 'left_leg_y', 'left_wing_x',
        'left_wing_y', 'nape_x', 'nape_y', 'right_cheek_x', 'right_cheek_y', 'right_eye_x', 'right_eye_y',
        'right_leg_x', 'right_leg_y', 'right_wing_x', 'right_wing_y', 'tail_x', 'tail_y', 'throat_x', 'throat_y']
        """
        with open(os.path.join(self.root, self.BASE_FOLDER, "images.txt"), "r") as f:
            reader = csv.DictReader(f, delimiter="\t")
            self.meta = list(reader)  # List of dictionaries.
        self.labels = [int(entry["species_id"]) - 1 for entry in self.meta]
        self.paths = [os.path.join(self.root, self.BASE_FOLDER, entry["path"]) for entry in self.meta]
        with open(os.path.join(self.root, self.BASE_FOLDER, "species.txt"), "r") as f:
            reader = csv.DictReader(f, delimiter="\t")
            self.classes_metadata = list(reader)
        self.classes = [str(entry["common"]) for entry in self.classes_metadata]
        f = open(os.path.join(self.root, 'birdsnap/test_images.txt'), "r")
        test_list = [line.split(',')[0] for line in f.read().splitlines()]
        new_path = []
        new_labels = []
        new_meta = []
        if self.split == 'train':
            for meta, file, label in zip(self.meta, self.paths, self.labels):
                if file.split('/')[-2] + '/' + file.split('/')[-1] not in test_list:
                    new_path.append(file)
                    new_labels.append(label)
                    new_meta.append(meta)
        else:
             for meta, file, label in zip(self.meta, self.paths, self.labels):
                if file.split('/')[-2] + '/' + file.split('/')[-1] in test_list:
                    new_path.append(file)
                    new_labels.append(label)
                    new_meta.append(meta)
        self.meta = new_meta
        self.paths = new_path
        self.labels = new_labels
        assert(len(self.meta)==len(self.paths)==len(self.labels))
        
    def _verify_image(self, idx):
        try:
            # Do this if you want to check in detail:
            # with open(os.path.join(self.root, self.BASE_FOLDER, self.meta[idx]['path']), 'rb') as fin:
            #     return (hashlib.md5(fin.read()).hexdigest() == self.meta[idx]['md5'])
            # In the mean time, just check if everything is there:
            return os.path.exists(os.path.join(self.root, self.BASE_FOLDER, self.meta[idx]["path"]))
        except FileNotFoundError:
            return False

    def scrape_images(self, missing_ids, chunk_size=8196):
        """Scrape images using the python default ThreadPool example."""
        import requests

        def _load_url_and_save_image(idx, timeout):
            full_path = os.path.join(self.root, self.BASE_FOLDER, self.meta[idx]["path"])
            os.makedirs(os.path.split(full_path)[0], exist_ok=True)
            r = requests.get(self.meta[idx]["url"], stream=True)
            with open(full_path, "wb") as write_file:
                for chunk in r.iter_content(chunk_size=chunk_size):
                    write_file.write(chunk)
            return True

        # We can use a with statement to ensure threads are cleaned up promptly
        with concurrent.futures.ThreadPoolExecutor(max_workers=None) as executor:  # Choose max_workers dynamically
            # Start the load operations and mark each future with its URL
            future_to_url = {
                executor.submit(_load_url_and_save_image, idx, 600): self.meta[idx]["url"] for idx in missing_ids
            }
            for future in concurrent.futures.as_completed(future_to_url):
                url = future_to_url[future]
                try:
                    data = future.result()
                except Exception as exc:
                    print(f"{url} generated exception: {exc}")
                else:
                    print(f"{url} downloaded successfully.")

    def _purge_missing_data(self):
        """Iterate over all data and throw out missing images."""
        JPG = b"\xff\xd8\xff"

        clean_meta = []
        invalid_files = 0
        for entry in self.meta:
            full_path = os.path.join(self.root, self.BASE_FOLDER, entry["path"])
            with open(full_path, "rb") as file_handle:
                if file_handle.read(3) == JPG:
                    clean_meta.append(entry)
                else:
                    invalid_files += 1
        print(f"Discarded {invalid_files} invalid files.")
        self.meta = clean_meta

        self.labels = [int(entry["species_id"]) - 1 for entry in self.meta]
        self.paths = [os.path.join(self.root, self.BASE_FOLDER, entry["path"]) for entry in self.meta]
