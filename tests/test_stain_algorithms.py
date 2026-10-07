"""Unit tests for stain normalization and augmentation algorithms."""

import unittest
import numpy as np

from src.config.constants import CLASSES, NUM_CLASSES, HE_STAIN_MATRIX
from src.config.experiments import get_all_experiments, filter_experiments, get_experiment_by_id
from src.normalization.reinhard import ReinhardNormalizer, rgb_to_lab, lab_to_rgb
from src.normalization.macenko import MacenkoNormalizer, macenko_fit, macenko_apply
from src.normalization.factory import NormalizerFactory
from src.augmentation.hed_jitter import hed_stain_jitter
from src.augmentation.policies import AugmentationPolicy


class TestStainAlgorithms(unittest.TestCase):
    def setUp(self):
        # Create synthetic H&E tile (100x100 RGB uint8) with distinct purple/pink shades
        np.random.seed(42)
        base = np.zeros((100, 100, 3), dtype=np.uint8)
        # Hematoxylin (nuclei, blue/purple)
        base[:50, :, :] = [130, 80, 160]
        # Eosin (stroma, pink)
        base[50:, :, :] = [220, 130, 170]
        # Add slight noise
        noise = np.random.randint(-10, 10, size=base.shape)
        self.sample_tile = np.clip(base.astype(np.int32) + noise, 0, 255).astype(np.uint8)

        # Reference tile
        ref = np.zeros((100, 100, 3), dtype=np.uint8)
        ref[:, :, :] = [180, 100, 150]
        self.ref_tile = ref

    def test_reinhard_normalizer(self):
        normalizer = ReinhardNormalizer()
        self.assertFalse(normalizer.is_fitted)
        normalizer.fit(self.ref_tile)
        self.assertTrue(normalizer.is_fitted)

        norm_img = normalizer.transform(self.sample_tile)
        self.assertEqual(norm_img.shape, self.sample_tile.shape)
        self.assertEqual(norm_img.dtype, np.uint8)
        self.assertTrue(0 <= norm_img.min() <= norm_img.max() <= 255)

    def test_reinhard_color_space_invertibility(self):
        lab = rgb_to_lab(self.sample_tile)
        reconstructed = lab_to_rgb(lab)
        # Reconstruction error should be minimal (within rounding/quantization)
        diff = np.abs(self.sample_tile.astype(int) - reconstructed.astype(int))
        self.assertLess(np.mean(diff), 2.0)

    def test_macenko_normalizer(self):
        normalizer = MacenkoNormalizer()
        self.assertFalse(normalizer.is_fitted)
        normalizer.fit(self.sample_tile)
        self.assertTrue(normalizer.is_fitted)

        norm_img = normalizer.transform(self.sample_tile)
        self.assertEqual(norm_img.shape, self.sample_tile.shape)
        self.assertEqual(norm_img.dtype, np.uint8)
        self.assertTrue(0 <= norm_img.min() <= norm_img.max() <= 255)

    def test_normalizer_factory(self):
        reinhard = NormalizerFactory.create("reinhard", reference_rgb=self.ref_tile)
        self.assertIsInstance(reinhard, ReinhardNormalizer)
        self.assertTrue(reinhard.is_fitted)

        macenko = NormalizerFactory.create("macenko", reference_rgb=self.sample_tile)
        self.assertIsInstance(macenko, MacenkoNormalizer)
        self.assertTrue(macenko.is_fitted)

        none_norm = NormalizerFactory.create("none")
        self.assertIsNone(none_norm)

        with self.assertRaises(ValueError):
            NormalizerFactory.create("invalid_norm_type")

    def test_hed_stain_jitter(self):
        jittered = hed_stain_jitter(self.sample_tile, sigma=0.2, bias=0.05)
        self.assertEqual(jittered.shape, self.sample_tile.shape)
        self.assertEqual(jittered.dtype, np.uint8)
        self.assertTrue(0 <= jittered.min() <= jittered.max() <= 255)

    def test_constants_and_experiments(self):
        self.assertEqual(len(CLASSES), 9)
        self.assertEqual(NUM_CLASSES, 9)

        all_exps = get_all_experiments()
        self.assertEqual(len(all_exps), 13)

        exp01 = get_experiment_by_id("EXP-01")
        self.assertIsNotNone(exp01)
        self.assertEqual(exp01.backbone, "resnet50")
        self.assertEqual(exp01.norm, "none")
        self.assertEqual(exp01.aug, "none")

        filtered = filter_experiments(["EXP-01", "EXP-07"])
        self.assertEqual(len(filtered), 2)
        self.assertEqual([e.id for e in filtered], ["EXP-01", "EXP-07"])

    def test_augmentation_policies(self):
        self.assertEqual(AugmentationPolicy.from_str("none"), AugmentationPolicy.NONE)
        self.assertEqual(AugmentationPolicy.from_str("aug_geo"), AugmentationPolicy.AUG_GEO)
        self.assertEqual(AugmentationPolicy.from_str("aug_stain"), AugmentationPolicy.AUG_STAIN)
        self.assertEqual(AugmentationPolicy.from_str("aug_combined"), AugmentationPolicy.AUG_COMBINED)


if __name__ == "__main__":
    unittest.main()
