import unittest
from model.hybrid_effres_attention import HybridModel

class TestHybridModel(unittest.TestCase):
    def test_model_output_shape(self):
        model = HybridModel(num_classes=3)
        input_tensor = torch.randn(1, 3, 256, 256)
        output = model(input_tensor)
        self.assertEqual(output.shape, (1, 3))

if __name__ == '__main__':
    unittest.main()
