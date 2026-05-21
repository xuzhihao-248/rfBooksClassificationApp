import torch
from src.transformer_model import SelfAttention, SimpleTransformer, PositionalEncoding


class TestPositionalEncoding:
    def test_shape(self):
        pe = PositionalEncoding(d_model=64, max_len=100)
        x = torch.randn(2, 50, 64)
        out = pe(x)
        assert out.shape == x.shape

    def test_adds_position_info(self):
        pe = PositionalEncoding(d_model=64)
        x = torch.zeros(1, 10, 64)
        out = pe(x)
        assert not torch.allclose(out[0, 0], out[0, 1])


class TestSelfAttention:
    def test_output_shape(self):
        attn = SelfAttention(d_model=64)
        x = torch.randn(4, 20, 64)
        out = attn(x)
        assert out.shape == x.shape

    def test_with_mask(self):
        attn = SelfAttention(d_model=64)
        x = torch.randn(2, 10, 64)
        mask = torch.zeros(2, 10, dtype=torch.bool)
        mask[:, 5:] = True
        out = attn(x, mask)
        assert out.shape == x.shape
        assert not torch.isnan(out).any()


class TestSimpleTransformer:
    def test_forward(self):
        vocab_size = 1000
        num_classes = 22
        model = SimpleTransformer(vocab_size, num_classes, d_model=64, max_len=128)
        x = torch.randint(0, vocab_size, (4, 30))
        lengths = torch.tensor([30, 25, 20, 15])
        logits = model(x, lengths)
        assert logits.shape == (4, num_classes)

    def test_padding_handling(self):
        vocab_size = 500
        model = SimpleTransformer(vocab_size, 22, d_model=32, max_len=64)
        x = torch.randint(1, vocab_size, (2, 20))
        x[:, 10:] = 0  # pad second half
        lengths = torch.tensor([10, 5])
        logits = model(x, lengths)
        assert logits.shape == (2, 22)
        assert not torch.isnan(logits).any()

    def test_gradient_flow(self):
        vocab_size = 500
        model = SimpleTransformer(vocab_size, 22, d_model=32, max_len=64)
        x = torch.randint(1, vocab_size, (2, 15))
        lengths = torch.tensor([15, 12])
        logits = model(x, lengths)
        loss = logits.sum()
        loss.backward()
        for name, param in model.named_parameters():
            assert param.grad is not None, f"{name} has no gradient"
            assert not torch.isnan(param.grad).any(), f"{name} has NaN gradient"
