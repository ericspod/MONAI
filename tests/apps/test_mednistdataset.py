# Copyright (c) MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

import shutil
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from monai.apps import MedNISTDataset, logger
from monai.data import MetaTensor
from monai.transforms import Compose, LoadImaged, ScaleIntensityd
from tests.test_utils import skip_if_downloading_fails, skip_if_quick

MEDNIST_FULL_DATASET_LENGTH = 58954


class TestMedNISTDataset(unittest.TestCase):
    def test_need_download(self):
        """Test that the dataset raises RuntimeError if the data isn't downloaded yet and `download` is False."""
        with TemporaryDirectory() as tdir, self.assertRaisesRegex(RuntimeError, "^Cannot find dataset directory"):
            MedNISTDataset(root_dir=tdir, section="test", download=False)

    def test_rootdir_not_dir(self):
        root_dir = Path("./NONEXISTENT")
        self.assertFalse(root_dir.is_dir())
        with self.assertRaisesRegex(ValueError, "Root directory root_dir must be a directory."):
            MedNISTDataset(root_dir=root_dir, section="test", download=False)

    @skip_if_quick
    def test_dataset_contents(self):
        """
        Test the dataset with an initial downloading of the tarball, followed by tests in subtests. This simplifies
        testing by downloading the data once and testing with it multiple ways in one simpler method.
        """
        testing_dir = Path(__file__).parents[1] / "testing_data"
        transform = Compose([LoadImaged(keys="image", ensure_channel_first=True), ScaleIntensityd(keys="image")])

        # common set of arguments for all dataset instantiations below
        common_args = dict(root_dir=testing_dir, section="test", progress=False)

        def _test_dataset(dataset: MedNISTDataset):
            self.assertEqual(dataset.get_num_classes(), 6)
            self.assertTupleEqual(dataset.get_classes(), ("AbdomenCT", "BreastMRI", "CXR", "ChestCT", "Hand", "HeadCT"))
            self.assertEqual(len(dataset), int(MEDNIST_FULL_DATASET_LENGTH * dataset.test_frac))
            self.assertTrue("image" in dataset[0])
            self.assertTrue("label" in dataset[0])
            self.assertIsInstance(dataset[0]["image"], MetaTensor)
            self.assertTupleEqual(dataset[0]["image"].shape, (1, 64, 64))

        with skip_if_downloading_fails():
            try:
                logger.disabled = True  # silence logging
                data = MedNISTDataset(transform=transform, download=True, copy_cache=False, **common_args)
                self.addCleanup(shutil.rmtree, str(testing_dir / "MedNIST"))  # cleanup expanded data when finished
            finally:
                logger.disabled = False

        with self.subTest("Test downloaded dataset properties"):
            _test_dataset(data)

        with self.subTest("Test non-downloaded dataset properties"):
            data = MedNISTDataset(transform=transform, download=False, runtime_cache=True, **common_args)
            _test_dataset(data)

        with self.subTest("Test dataset without transforms"):
            data = MedNISTDataset(download=False, **common_args)
            self.assertTupleEqual(data[0]["image"].shape, (64, 64))

        with self.subTest("Test with different random seed"):
            data = MedNISTDataset(transform=transform, download=False, seed=42, **common_args)
            _test_dataset(data)
            self.assertEqual(data[0]["class_name"], "AbdomenCT")
            self.assertEqual(data[0]["label"], 0)


if __name__ == "__main__":
    unittest.main()
