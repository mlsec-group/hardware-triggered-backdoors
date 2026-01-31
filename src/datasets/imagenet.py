from pathlib import Path

from torchvision.datasets import ImageFolder
from torchvision.transforms import transforms

from datasets.common import DatasetLoader

try:
    import nvidia.dali.fn as fn
    import nvidia.dali.types as types
    from nvidia.dali.pipeline import Pipeline
    from nvidia.dali.plugin.base_iterator import LastBatchPolicy
    from nvidia.dali.plugin.pytorch import DALIGenericIterator

except ImportError:
    pass


class ImageNetLoader(DatasetLoader):
    def __init__(self, dataset_dir: Path):
        dataset_dir = Path(dataset_dir) if isinstance(dataset_dir, str) else dataset_dir
        self.dataset_dir = dataset_dir

        self.normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )

        self.train_transform = transforms.Compose(
            [
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                # normalize,
            ]
        )
        self.val_transform = transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                # normalize,
            ]
        )

    def normalize(self, X):
        return self.normalize(X)

    def load_train(self):
        return ImageFolder(
            root=self.dataset_dir / "train_set", transform=self.train_transform
        )

    def load_train_deterministic(self):
        return ImageFolder(
            root=self.dataset_dir / "train_set", transform=self.val_transform
        )

    def load_test(self):
        return ImageFolder(
            root=self.dataset_dir / "val_set", transform=self.val_transform
        )

    def input_shape(self):
        return (3, 224, 224)

    def output_shape(self):
        return (1000,)


class ImageNetAccuracyEvaluator:
    def __init__(
        self,
        dataset_path,
        batch_size=256,
        num_threads=4,
        device_id=0,
        random_shuffle=False,
    ):
        self.dataset_path = dataset_path
        self.batch_size = batch_size
        self.num_threads = num_threads
        self.device_id = device_id
        self.random_shuffle = random_shuffle

    def _create_pipeline(self):
        """
        Creates a DALI pipeline to perform:
        - reading imagenet val images
        - jpeg decoding (gpu/mixed)
        - resizing
        - center crop
        - normalization
        """
        pipe = Pipeline(
            batch_size=self.batch_size,
            num_threads=self.num_threads,
            device_id=self.device_id,
        )

        with pipe:
            # Read files
            jpegs, labels = fn.readers.file(
                file_root=self.dataset_path,
                random_shuffle=self.random_shuffle,
                name="Reader",
            )

            images = fn.decoders.image(jpegs, device="mixed", output_type=types.RGB)

            images = fn.resize(images, resize_shorter=256)

            images = fn.crop_mirror_normalize(
                images,
                dtype=types.FLOAT,
                crop=(224, 224),
                mean=[0.485 * 255, 0.456 * 255, 0.406 * 255],
                std=[0.229 * 255, 0.224 * 255, 0.225 * 255],
            )

            # DALI pipelines must return lists
            pipe.set_outputs(images, labels)

        return pipe

    def get_iterator(self):
        """
        Runs inference using the GPU-accelerated NVIDIA DALI pipeline
        and computes accuracy over the Imagenet validation set.
        """
        pipe = self._create_pipeline()
        pipe.build()

        dali_iter = DALIGenericIterator(
            pipelines=pipe,
            output_map=["data", "label"],
            auto_reset=False,
            last_batch_policy=LastBatchPolicy.PARTIAL,
        )

        return dali_iter
