import os
import pytest
os.chdir(os.path.dirname(__file__))


def test_CoordRegress():
    """Test coordinate regression layer"""
    import torch
    from network.seg_net_1d import CoordRegress

    a = torch.tensor([[[0,   0, 1/3, 1/3, 1/3],
                       [1/5, 0,   0, 4/5,   0]]], dtype=torch.float32)
    m = CoordRegress(length=5)
    c = m(a).numpy()[0]

    assert pytest.approx(5 * c[0]) == 3
    assert pytest.approx(5 * c[1]) == 2.4


def test_variance_regularizer():
    """Test heatmap variance calculation"""
    import torch
    from network.seg_net_1d import SegNet1D

    size = 256
    loc = 0.25
    var = 0.005
    c = torch.tensor([[loc]], dtype=torch.float32)
    p = torch.arange(size, dtype=torch.float32) / size
    p = p.view(1, 1, size)
    p = torch.exp(-0.5*(p-loc)**2/var)
    p = p / torch.sum(p)
    m = SegNet1D(input_size=(size, 16))
    reg = m.regularizer(p, c)
    reg = reg.item()
    assert pytest.approx(reg, rel=1e-2) == var
    # Var is noisy due to discrete coords and tail cutoff


def test_SegNet1D_shape():
    """Test input and output shape"""
    import torch
    from network.seg_net_1d import SegNet1D

    a = torch.ones((1, 1, 1024, 32), dtype=torch.float32)
    b = torch.ones((1, 1, 1024, 256), dtype=torch.float32)
    c = torch.ones((1, 1, 1040, 256), dtype=torch.float32)
    m = SegNet1D()
    output_a, _ = m(a)
    output_b, _ = m(b)

    assert output_a.shape == (1, 2,)
    assert output_b.shape == (1, 2,)
    with pytest.raises(RuntimeError):
        m(c)


def test_SegNet1D_full_img_mode():
    """Test input and output shape"""
    import torch
    from network.seg_net_1d import SegNet1D

    a = torch.ones((1, 1, 1024, 256), dtype=torch.float32)
    m = SegNet1D()
    m.full_image_mode = True
    coords, heatmap = m(a)

    assert coords.shape == (1, 2, 256)
    assert heatmap.shape == (1, 2, 1024, 256)


def test_SegNet1D_changes():
    """Test that all trainable parameters update on training"""
    import torch
    from network.seg_net_1d import SegNet1D

    x = torch.randn((1, 1, 1024, 32), dtype=torch.float32)
    y = torch.randint(300, 700, (1, 2), dtype=torch.float32)
    m = SegNet1D()
    if torch.cuda.is_available():
        x = x.cuda()
        y = y.cuda()
        m.cuda()
    params = [p for p in m.named_parameters() if p[1].requires_grad]
    old_params = [(name, p.clone()) for (name, p) in params]
    m.train()

    optimizer = torch.optim.SGD(m.parameters(), lr=1.0)
    optimizer.zero_grad()
    output, regularizer = m(x)
    loss = torch.nn.MSELoss()(output, y) + regularizer
    loss.backward()
    optimizer.step()

    for (name, p0), (_, p1) in zip(old_params, params):
        try:
            assert not torch.equal(p0, p1)
        except AssertionError:
            raise AssertionError(name + ' did not change')
